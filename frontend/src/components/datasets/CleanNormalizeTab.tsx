import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ChevronDown, ChevronRight, Sparkles, Eraser } from 'lucide-react'
import { toast } from 'sonner'
import { DatasetSelector } from './DatasetSelector'

interface OperationResult {
  rows_before?: number
  rows_after?: number
  [key: string]: unknown
}

export function CleanNormalizeTab() {
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()

  const [selectedDataset, setSelectedDataset] = useState('')
  const [showCleanControls, setShowCleanControls] = useState(false)
  const [showNormControls, setShowNormControls] = useState(false)
  const [lastResult, setLastResult] = useState<OperationResult | null>(null)

  async function runTool(toolName: string) {
    if (!selectedDataset) return
    try {
      const result = await executeTool({
        toolName,
        args: { dataset_path: selectedDataset },
      })
      const opResult = result as unknown as OperationResult
      setLastResult(opResult)
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success(`${toolName} completed`)
    } catch (err) {
      toast.error(`${toolName} failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <DatasetSelector
        value={selectedDataset}
        onChange={setSelectedDataset}
        label="Select dataset to process"
      />

      {lastResult && (lastResult.rows_before != null || lastResult.rows_after != null) && (
        <div className="flex items-center gap-3">
          {lastResult.rows_before != null && (
            <Badge variant="secondary">Before: {lastResult.rows_before} rows</Badge>
          )}
          {lastResult.rows_after != null && (
            <Badge variant="success">After: {lastResult.rows_after} rows</Badge>
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
            <CardDescription>Remove duplicates, validate schema, filter short entries</CardDescription>
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
      </div>
    </div>
  )
}
