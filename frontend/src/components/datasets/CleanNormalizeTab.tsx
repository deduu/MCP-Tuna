import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { buildDatasetOutputPath } from '@/lib/dataset-output'
import { ChevronDown, ChevronRight, Sparkles, Eraser } from 'lucide-react'
import { toast } from 'sonner'
import { DatasetSelector } from './DatasetSelector'

interface OperationResult {
  tool_name?: string
  rows_before?: number
  rows_after?: number
  output_path?: string
  [key: string]: unknown
}

const REMAP_PRESET_OPTIONS = [
  {
    value: 'chat_triplet_to_sft',
    label: 'Chat Triplet -> SFT',
    description: 'Maps system/user/assistant into instruction/input/output rows for direct SFT training.',
  },
  {
    value: 'prompt_response_to_sft',
    label: 'Prompt/Response -> SFT',
    description: 'Maps prompt/response datasets into instruction/input/output rows.',
  },
  {
    value: 'qa_to_sft',
    label: 'Q/A -> SFT',
    description: 'Maps question/answer datasets into instruction/input/output rows.',
  },
] as const

export function CleanNormalizeTab() {
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()

  const [selectedDataset, setSelectedDataset] = useState('')
  const [showCleanControls, setShowCleanControls] = useState(false)
  const [showNormControls, setShowNormControls] = useState(false)
  const [remapPreset, setRemapPreset] = useState<(typeof REMAP_PRESET_OPTIONS)[number]['value']>('chat_triplet_to_sft')
  const [lastResult, setLastResult] = useState<OperationResult | null>(null)

  function toNumber(value: unknown, fallback = 0): number {
    const num = Number(value)
    return Number.isFinite(num) ? num : fallback
  }

  function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === 'object' ? (value as Record<string, unknown>) : null
  }

  function buildInsights(result: OperationResult): string[] {
    const toolName = typeof result.tool_name === 'string' ? result.tool_name : ''
    const steps = asRecord(result.steps)
    const insights: string[] = []

    if (toolName === 'clean.dataset') {
      const empty = asRecord(steps?.remove_empty_fields)
      const dedupe = asRecord(steps?.deduplicate)
      const short = asRecord(steps?.remove_short_entries)
      const removed = toNumber(result.removed)

      if (removed === 0) {
        insights.push(
          'No rows were removed. The dataset passed empty-field checks, deduplication, and short-entry filtering as-is.',
        )
      }
      if (empty?.enabled !== false) {
        insights.push(`Empty-field check removed ${toNumber(empty?.removed)} row(s).`)
      }
      if (dedupe?.enabled !== false) {
        insights.push(`Deduplication removed ${toNumber(dedupe?.removed)} duplicate row(s).`)
      }
      if (short?.enabled !== false) {
        insights.push(
          `Short-entry filter removed ${toNumber(short?.removed)} row(s) using min instruction ${toNumber(short?.min_instruction)} and min output ${toNumber(short?.min_output)} characters.`,
        )
      }
      return insights
    }

    if (toolName === 'clean.deduplicate') {
      insights.push(`Deduplication removed ${toNumber(result.duplicates_removed)} duplicate row(s).`)
      if (toNumber(result.duplicates_removed) === 0) {
        insights.push('No duplicate instruction values were found.')
      }
      return insights
    }

    if (toolName === 'clean.validate_schema') {
      const requiredFields = Array.isArray(result.required_fields)
        ? result.required_fields.map(String).join(', ')
        : 'instruction, output'
      insights.push(`Required fields checked: ${requiredFields}.`)
      insights.push(
        `Valid rows: ${toNumber(result.valid_count)}. Invalid rows: ${toNumber(result.invalid_count)}.`,
      )
      return insights
    }

    if (toolName === 'clean.remove_short') {
      insights.push(`Short-entry filter removed ${toNumber(result.removed)} row(s).`)
      insights.push(
        `Thresholds used: instruction >= ${toNumber(result.min_instruction ?? result.min_instruction_length, 10)}, output >= ${toNumber(result.min_output ?? result.min_output_length, 20)} characters.`,
      )
      return insights
    }

    if (toolName === 'normalize.dataset') {
      const strip = asRecord(steps?.strip_text)
      const merge = asRecord(steps?.merge_fields)
      const standardize = asRecord(steps?.standardize_keys)
      const changedRows = toNumber(result.changed_rows)

      if (changedRows === 0) {
        insights.push(
          'No rows changed. Text cleanup found nothing to trim, no instruction/input pairs were merged, and no alternate keys needed renaming.',
        )
      } else {
        insights.push(`Normalization changed ${changedRows} row(s) overall.`)
      }
      if (strip?.enabled !== false) {
        insights.push(
          `Text cleanup changed ${toNumber(strip?.changed_rows)} row(s) across ${toNumber(strip?.changed_fields)} field(s).`,
        )
      }
      if (merge?.enabled !== false) {
        insights.push(`Field merge combined instruction + input in ${toNumber(merge?.merged_rows)} row(s).`)
      }
      if (standardize?.enabled !== false) {
        insights.push(
          `Key standardization renamed ${toNumber(standardize?.renamed_fields)} field name(s) for ${String(standardize?.target_format ?? result.target_format ?? 'sft')} format.`,
        )
      }
      return insights
    }

    if (toolName === 'normalize.merge_fields') {
      insights.push(`Field merge combined instruction + input in ${toNumber(result.merged_rows)} row(s).`)
      if (toNumber(result.merged_rows) === 0) {
        insights.push('No non-empty input fields were available to merge.')
      }
      return insights
    }

    if (toolName === 'normalize.standardize_keys') {
      insights.push(
        `Key standardization renamed ${toNumber(result.renamed_fields)} field name(s) for ${String(result.target_format ?? 'sft')} format.`,
      )
      if (toNumber(result.renamed_fields) === 0) {
        insights.push('The dataset already used the expected field names.')
      }
      return insights
    }

    if (toolName === 'normalize.remap_fields') {
      insights.push(
        `Schema conversion changed ${toNumber(result.changed_rows)} row(s) using preset ${String(result.preset ?? 'unknown')}.`,
      )
      const createdFields = Array.isArray(result.created_fields) ? result.created_fields.map(String).join(', ') : ''
      const droppedFields = Array.isArray(result.dropped_fields) ? result.dropped_fields.map(String).join(', ') : ''
      if (createdFields) {
        insights.push(`Created fields: ${createdFields}.`)
      }
      if (droppedFields) {
        insights.push(`Dropped source fields: ${droppedFields}.`)
      }
      insights.push(`Converted rows are ready for ${String(result.target_format ?? 'sft').toUpperCase()} training.`)
      return insights
    }

    if (toolName === 'normalize.strip_text') {
      insights.push(
        `Text cleanup changed ${toNumber(result.changed_rows)} row(s) across ${toNumber(result.changed_fields)} field(s).`,
      )
      if (toNumber(result.changed_rows) === 0) {
        insights.push('No leading/trailing whitespace or unicode normalization fixes were needed.')
      }
      return insights
    }

    return insights
  }

  function buildOutputPath(filePath: string, toolName: string): string {
    const shortTool = toolName.split('.').pop() ?? 'processed'
    return buildDatasetOutputPath(filePath, shortTool)
  }

  async function runTool(
    toolName: string,
    extraArgs: Record<string, unknown> = {},
    outputSuffix?: string,
  ) {
    if (!selectedDataset) return
    try {
      const loaded = await executeTool({
        toolName: 'dataset.load',
        args: { file_path: selectedDataset },
      })
      const loadedObj = loaded as Record<string, unknown>
      const dataPoints = Array.isArray(loadedObj.data_points)
        ? (loadedObj.data_points as Array<Record<string, unknown>>)
        : []
      if (dataPoints.length === 0) {
        toast.error('Dataset has no rows to process')
        return
      }

      const result = await executeTool({
        toolName,
        args: { data_points: dataPoints, ...extraArgs },
      })
      const resultObj = result as Record<string, unknown>
      if (resultObj.success === false) {
        throw new Error(
          typeof resultObj.error === 'string' && resultObj.error.trim()
            ? resultObj.error
            : `${toolName} failed`,
        )
      }
      const processed = Array.isArray(resultObj.data_points)
        ? (resultObj.data_points as Array<Record<string, unknown>>)
        : []
      const outputPath = buildOutputPath(selectedDataset, outputSuffix ?? toolName)

      if (processed.length > 0) {
        await executeTool({
          toolName: 'dataset.save',
          args: { data_points: processed, output_path: outputPath, format: 'jsonl' },
        })
      }

      const rowsBefore = Number(resultObj.original_count ?? dataPoints.length)
      const rowsAfter = Number(
        resultObj.cleaned_count ?? resultObj.count ?? resultObj.filtered_count ?? processed.length,
      )
      const opResult = {
        ...resultObj,
        rows_before: Number.isFinite(rowsBefore) ? rowsBefore : dataPoints.length,
        rows_after: Number.isFinite(rowsAfter) ? rowsAfter : processed.length,
        output_path: outputPath,
        tool_name: toolName,
      } as OperationResult
      setLastResult(opResult)
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success(`${toolName} completed and saved`)
    } catch (err) {
      toast.error(`${toolName} failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  const insightLines = lastResult ? buildInsights(lastResult) : []

  return (
    <div className="space-y-6 max-w-4xl">
      <DatasetSelector
        value={selectedDataset}
        onChange={setSelectedDataset}
        label="Select dataset to process"
      />

      {lastResult && (lastResult.rows_before != null || lastResult.rows_after != null) && (
        <div className="space-y-3">
          <div className="flex items-center gap-3 flex-wrap">
            {lastResult.rows_before != null && (
              <Badge variant="secondary">Before: {lastResult.rows_before} rows</Badge>
            )}
            {lastResult.rows_after != null && (
              <Badge variant="success">After: {lastResult.rows_after} rows</Badge>
            )}
            {lastResult.output_path && (
              <Badge variant="outline">Saved: {lastResult.output_path.split(/[\\/]/).pop()}</Badge>
            )}
          </div>
          {insightLines.length > 0 && (
            <div className="rounded-md border border-border/60 bg-secondary/20 p-3">
              <p className="text-sm font-medium">What happened</p>
              <div className="mt-2 space-y-1">
                {insightLines.map((line) => (
                  <p key={line} className="text-xs text-muted-foreground">
                    {line}
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Clean Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Eraser className="h-4 w-4" />
              Clean
            </CardTitle>
            <CardDescription>Remove empty fields, duplicates, and short entries</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button
              onClick={() => runTool('clean.dataset')}
              disabled={isPending || !selectedDataset}
            >
              Full Clean
            </Button>

            <div>
              <button
                onClick={() => setShowCleanControls(!showCleanControls)}
                className="flex items-center gap-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              >
                {showCleanControls ? (
                  <ChevronDown className="h-3.5 w-3.5" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5" />
                )}
                Fine-grained Controls
              </button>

              {showCleanControls && (
                <div className="mt-3 space-y-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => runTool('clean.deduplicate')}
                    disabled={isPending || !selectedDataset}
                  >
                    Deduplicate
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => runTool('clean.validate_schema')}
                    disabled={isPending || !selectedDataset}
                    className="ml-2"
                  >
                    Validate Schema
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => runTool('clean.remove_short')}
                    disabled={isPending || !selectedDataset}
                    className="ml-2"
                  >
                    Remove Short
                  </Button>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Normalize Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Sparkles className="h-4 w-4" />
              Normalize
            </CardTitle>
            <CardDescription>Merge fields, standardize keys, strip text</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button
              onClick={() => runTool('normalize.dataset')}
              disabled={isPending || !selectedDataset}
            >
              Full Normalize
            </Button>

            <div>
              <button
                onClick={() => setShowNormControls(!showNormControls)}
                className="flex items-center gap-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              >
                {showNormControls ? (
                  <ChevronDown className="h-3.5 w-3.5" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5" />
                )}
                Fine-grained Controls
              </button>

              {showNormControls && (
                <div className="mt-3 space-y-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => runTool('normalize.merge_fields')}
                    disabled={isPending || !selectedDataset}
                  >
                    Merge Fields
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => runTool('normalize.standardize_keys')}
                    disabled={isPending || !selectedDataset}
                    className="ml-2"
                  >
                    Standardize Keys
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => runTool('normalize.strip_text')}
                    disabled={isPending || !selectedDataset}
                    className="ml-2"
                  >
                    Strip Text
                  </Button>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Sparkles className="h-4 w-4" />
              Convert Schema
            </CardTitle>
            <CardDescription>Turn chat or QA datasets into instruction/input/output rows you can train on directly</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1">
              <label className="text-sm font-medium">Preset</label>
              <select
                value={remapPreset}
                onChange={(e) => setRemapPreset(e.target.value as (typeof REMAP_PRESET_OPTIONS)[number]['value'])}
                className="w-full h-9 rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                disabled={isPending || !selectedDataset}
              >
                {REMAP_PRESET_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <p className="text-xs text-muted-foreground">
              {REMAP_PRESET_OPTIONS.find((option) => option.value === remapPreset)?.description}
            </p>
            <Button
              onClick={() => runTool('normalize.remap_fields', { preset: remapPreset }, `remap_${remapPreset}`)}
              disabled={isPending || !selectedDataset}
            >
              Convert And Save
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
