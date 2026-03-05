import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { toast } from 'sonner'
import { MetricsTable } from './MetricsTable'
import { ExportButton } from './ExportButton'

export function FtEvalTab() {
  const [modelPath, setModelPath] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [compareModelPath, setCompareModelPath] = useState('')
  const [evalResult, setEvalResult] = useState<Record<string, unknown> | null>(null)
  const [detailedMetrics, setDetailedMetrics] = useState<Record<string, unknown> | null>(null)
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null)
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [loadingCompare, setLoadingCompare] = useState(false)

  async function handleEvaluate() {
    if (!modelPath.trim() || !datasetPath.trim()) {
      toast.error('Model path and dataset path are required')
      return
    }
    try {
      const res = await executeTool({
        toolName: 'ft_eval.evaluate_finetune',
        args: { model_path: modelPath, dataset_path: datasetPath },
      })
      setEvalResult(res as Record<string, unknown>)
      toast.success('Evaluation complete')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Evaluation failed')
    }
  }

  async function handleDetailedMetrics() {
    if (!modelPath.trim()) {
      toast.error('Model path is required')
      return
    }
    setLoadingDetail(true)
    try {
      const res = await executeTool({
        toolName: 'ft_eval.get_metrics',
        args: { model_path: modelPath },
      })
      setDetailedMetrics(res as Record<string, unknown>)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to get metrics')
    } finally {
      setLoadingDetail(false)
    }
  }

  async function handleCompare() {
    if (!modelPath.trim() || !compareModelPath.trim()) {
      toast.error('Both model paths are required for comparison')
      return
    }
    setLoadingCompare(true)
    try {
      const res = await executeTool({
        toolName: 'ft_eval.compare_models',
        args: { model_path_a: modelPath, model_path_b: compareModelPath },
      })
      setCompareResult(res as Record<string, unknown>)
      toast.success('Comparison complete')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Comparison failed')
    } finally {
      setLoadingCompare(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="pt-6 space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">Model Path</label>
            <Input
              placeholder="/path/to/fine-tuned-model"
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
            {isPending ? 'Running...' : 'Run Evaluation'}
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
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleDetailedMetrics}
                disabled={loadingDetail}
              >
                {loadingDetail ? 'Loading...' : 'View Detailed Metrics'}
              </Button>
              <ExportButton
                toolName="ft_eval.export_results"
                args={{ model_path: modelPath }}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {detailedMetrics && (
        <Card>
          <CardHeader>
            <CardTitle>Detailed Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <MetricsTable data={detailedMetrics} />
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Compare Models</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">Second Model Path</label>
            <Input
              placeholder="/path/to/second-model"
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
