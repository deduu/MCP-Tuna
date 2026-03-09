import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { toast } from 'sonner'
import { MetricsTable } from './MetricsTable'

export function FtEvalTab() {
  const [modelPath, setModelPath] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [compareModelPath, setCompareModelPath] = useState('')
  const [exportPath, setExportPath] = useState('output/ft_eval_results.jsonl')
  const [evalResult, setEvalResult] = useState<Record<string, unknown> | null>(null)
  const [summaryResult, setSummaryResult] = useState<Record<string, unknown> | null>(null)
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null)
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
      const rows = [...testData]
      if (modelPath.trim() && rows.length > 0) {
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

      const res = await executeTool({
        toolName: 'ft_eval.batch',
        args: { test_data: rows },
      })
      setEvalResult(res as Record<string, unknown>)
      setSummaryResult(null)
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
        toolName: 'ft_eval.export',
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
            <Input
              placeholder="/path/to/model"
              value={modelPath}
              onChange={(e) => setModelPath(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-medium">Dataset Path</label>
            <Input
              placeholder="/path/to/eval-dataset.jsonl"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
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
          </CardHeader>
          <CardContent className="space-y-4">
            <MetricsTable data={evalResult} />
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleSummary}
                disabled={loadingSummary}
              >
                {loadingSummary ? 'Loading...' : 'Compute Summary'}
              </Button>
              <Input
                className="max-w-sm"
                placeholder="output/ft_eval_results.jsonl"
                value={exportPath}
                onChange={(e) => setExportPath(e.target.value)}
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
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">Fine-tuned Adapter Path</label>
            <Input
              placeholder="/path/to/adapter"
              value={compareModelPath}
              onChange={(e) => setCompareModelPath(e.target.value)}
            />
          </div>
          <Button
            variant="outline"
            onClick={handleCompare}
            disabled={loadingCompare}
          >
            {loadingCompare ? 'Comparing...' : 'Compare'}
          </Button>
          {compareResult && <MetricsTable data={compareResult} />}
        </CardContent>
      </Card>
    </div>
  )
}
