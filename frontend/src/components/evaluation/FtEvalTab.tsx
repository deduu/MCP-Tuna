import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { toast } from 'sonner'
import { MetricsTable } from './MetricsTable'
import { BrowsePathField } from './BrowsePathField'
import { ModelPathField } from '@/components/pipeline/ModelPathField'
import { extractAssistantText, extractPromptText, isVlmDatasetRow } from '@/lib/evaluation-multimodal'

type EvaluationFlavor = 'text' | 'vlm' | null

export function FtEvalTab() {
  const [modelPath, setModelPath] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [compareModelPath, setCompareModelPath] = useState('')
  const [exportPath, setExportPath] = useState('output/ft_eval_results.jsonl')
  const [evalResult, setEvalResult] = useState<Record<string, unknown> | null>(null)
  const [summaryResult, setSummaryResult] = useState<Record<string, unknown> | null>(null)
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null)
  const [evaluationFlavor, setEvaluationFlavor] = useState<EvaluationFlavor>(null)
  const [isVlmDataset, setIsVlmDataset] = useState(false)
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const [loadingSummary, setLoadingSummary] = useState(false)
  const [loadingCompare, setLoadingCompare] = useState(false)
  const [exporting, setExporting] = useState(false)

  async function loadDataPoints(filePath: string) {
    const loaded = await executeTool({
      toolName: 'dataset.load',
      args: { file_path: filePath },
    })
    const payload = loaded as Record<string, unknown>
    return Array.isArray(payload.data_points)
      ? (payload.data_points as Array<Record<string, unknown>>)
      : []
  }

  async function handleEvaluate() {
    if (!datasetPath.trim()) {
      toast.error('Dataset path is required')
      return
    }
    try {
      const testData = await loadDataPoints(datasetPath)
      const rows = testData.map((row) => ({ ...row }))
      const vlmDataset = rows.some((row) => isVlmDatasetRow(row))
      setIsVlmDataset(vlmDataset)

      if (modelPath.trim() && rows.length > 0) {
        if (vlmDataset) {
          for (const row of rows) {
            if (!isVlmDatasetRow(row)) {
              continue
            }
            const rowRecord = row as Record<string, unknown>
            const inference = await executeTool({
              toolName: 'test.vlm_inference',
              args: { messages: rowRecord.messages, model_path: modelPath.trim() },
            })
            rowRecord.generated = (inference as Record<string, unknown>).response ?? rowRecord.generated ?? ''
            rowRecord.reference = rowRecord.reference ?? extractAssistantText(rowRecord.messages)
          }
        } else {
          const prompts = rows.map((r) => String(r.instruction ?? r.prompt ?? ''))
          const inf = await executeTool({
            toolName: 'test.inference',
            args: { prompts, model_path: modelPath.trim() },
          })
          const infRows = Array.isArray((inf as Record<string, unknown>).results)
            ? ((inf as Record<string, unknown>).results as Array<Record<string, unknown>>)
            : []
          rows.forEach((row, i) => {
            row.generated = infRows[i]?.response ?? row.generated ?? ''
            row.reference = row.reference ?? row.output ?? ''
          })
        }
      }

      if (vlmDataset) {
        rows.forEach((row) => {
          if (!isVlmDatasetRow(row)) {
            return
          }
          const rowRecord = row as Record<string, unknown>
          rowRecord.reference = rowRecord.reference ?? extractAssistantText(rowRecord.messages)
          rowRecord.prompt = rowRecord.prompt ?? extractPromptText(rowRecord.messages)
        })
      }

      const toolName = vlmDataset ? 'judge.evaluate_vlm_batch' : 'ft_eval.batch'
      const res = await executeTool({ toolName, args: { test_data: rows } })
      const parsed = res as Record<string, unknown>
      setEvalResult(parsed)
      setSummaryResult(
        parsed.summary && typeof parsed.summary === 'object'
          ? (parsed.summary as Record<string, unknown>)
          : null,
      )
      setEvaluationFlavor(vlmDataset ? 'vlm' : 'text')
      toast.success('Evaluation complete')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Evaluation failed')
    }
  }

  async function handleSummary() {
    if (!evalResult || !Array.isArray(evalResult.results)) {
      toast.error('Run evaluation first')
      return
    }
    if (evaluationFlavor === 'vlm') {
      setSummaryResult(
        evalResult.summary && typeof evalResult.summary === 'object'
          ? (evalResult.summary as Record<string, unknown>)
          : null,
      )
      return
    }
    setLoadingSummary(true)
    try {
      const res = await executeTool({
        toolName: 'ft_eval.summary',
        args: { results: evalResult.results },
      })
      setSummaryResult(res as Record<string, unknown>)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to compute summary')
    } finally {
      setLoadingSummary(false)
    }
  }

  async function handleCompare() {
    if (isVlmDataset) {
      toast.error('Compare Models currently supports text datasets only')
      return
    }
    if (!modelPath.trim() || !compareModelPath.trim() || !datasetPath.trim()) {
      toast.error('Base model, adapter/model path, and dataset path are required')
      return
    }
    setLoadingCompare(true)
    try {
      const testData = await loadDataPoints(datasetPath)
      const prompts = testData.slice(0, 20).map((r) => String(r.instruction ?? r.prompt ?? ''))
      const res = await executeTool({
        toolName: 'test.compare_models',
        args: {
          prompts,
          base_model_path: modelPath.trim(),
          finetuned_adapter_path: compareModelPath.trim(),
        },
      })
      setCompareResult(res as Record<string, unknown>)
      toast.success('Comparison complete')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Comparison failed')
    } finally {
      setLoadingCompare(false)
    }
  }

  async function handleExport() {
    if (!evalResult || !Array.isArray(evalResult.results)) {
      toast.error('No FT eval results to export')
      return
    }
    setExporting(true)
    try {
      await executeTool({
        toolName: evaluationFlavor === 'vlm' ? 'judge.export' : 'ft_eval.export',
        args: { results: evalResult.results, output_path: exportPath, format: 'jsonl' },
      })
      toast.success(`Exported to ${exportPath}`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="pt-6 space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">Model Path (optional for inference)</label>
            <ModelPathField
              value={modelPath}
              onChange={setModelPath}
              placeholder="/path/to/model"
              helperText="Use a Hugging Face model ID or browse a backend-visible model folder."
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-medium">Dataset Path</label>
            <BrowsePathField
              value={datasetPath}
              onChange={setDatasetPath}
              placeholder="/path/to/eval-dataset.jsonl"
              allowFiles
              allowDirectories={false}
              preferredRootIds={['workspace', 'uploads', 'output']}
              helperText="Browse a dataset file visible to the gateway. Canonical VLM rows with messages will automatically use multimodal inference and judging."
            />
          </div>
          <Button onClick={handleEvaluate} disabled={isPending}>
            {isPending ? 'Running...' : 'Run FT Evaluation'}
          </Button>
        </CardContent>
      </Card>

      {evalResult && (
        <Card>
          <CardHeader>
            <CardTitle>Evaluation Results</CardTitle>
            {evaluationFlavor === 'vlm' && (
              <CardDescription>
                This result came from `test.vlm_inference` plus `judge.evaluate_vlm_batch`.
              </CardDescription>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            <MetricsTable data={evalResult} />
            <div className="space-y-3">
              <Button
                variant="outline"
                size="sm"
                onClick={handleSummary}
                disabled={loadingSummary || (evaluationFlavor === 'vlm' && !evalResult.summary)}
              >
                {loadingSummary ? 'Loading...' : evaluationFlavor === 'vlm' ? 'Show Summary' : 'Compute Summary'}
              </Button>
              <BrowsePathField
                value={exportPath}
                onChange={setExportPath}
                placeholder="output/ft_eval_results.jsonl"
                allowFiles
                allowDirectories
                preferredRootIds={['output', 'workspace']}
                directorySelectionMode="append-filename"
                defaultFileName="ft_eval_results.jsonl"
                helperText="Pick an existing export file or browse to a folder and keep the filename."
              />
              <Button variant="outline" size="sm" onClick={handleExport} disabled={exporting}>
                {exporting ? 'Exporting...' : 'Export'}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {summaryResult && (
        <Card>
          <CardHeader>
            <CardTitle>Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <MetricsTable data={summaryResult} />
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Compare Models (test.compare_models)</CardTitle>
          {isVlmDataset && (
            <CardDescription>
              Model-to-model comparison still uses the text inference tool. Use Judge Compare for multimodal side-by-side evaluation.
            </CardDescription>
          )}
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">Fine-tuned Adapter Path</label>
            <ModelPathField
              value={compareModelPath}
              onChange={setCompareModelPath}
              placeholder="/path/to/adapter"
              validationPurpose="adapter"
              helperText="Browse the fine-tuned adapter folder or enter a backend-visible path."
            />
          </div>
          <Button
            variant="outline"
            onClick={handleCompare}
            disabled={loadingCompare || isVlmDataset}
          >
            {loadingCompare ? 'Comparing...' : 'Compare'}
          </Button>
          {compareResult && <MetricsTable data={compareResult} />}
        </CardContent>
      </Card>
    </div>
  )
}
