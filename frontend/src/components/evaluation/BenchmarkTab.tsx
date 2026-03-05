import { useState } from 'react'
import { Plus, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { toast } from 'sonner'
import { MetricsTable } from './MetricsTable'
import { ExportButton } from './ExportButton'

export function BenchmarkTab() {
  const [modelPaths, setModelPaths] = useState<string[]>([''])
  const [datasetPath, setDatasetPath] = useState('')
  const [benchmarkResult, setBenchmarkResult] = useState<Record<string, unknown> | null>(null)
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null)
  const { mutateAsync: executeTool } = useToolExecution()
  const [runningBenchmark, setRunningBenchmark] = useState(false)
  const [runningCompare, setRunningCompare] = useState(false)

  function addModel() {
    setModelPaths((prev) => [...prev, ''])
  }

  function removeModel(index: number) {
    setModelPaths((prev) => prev.filter((_, i) => i !== index))
  }

  function updateModel(index: number, value: string) {
    setModelPaths((prev) => prev.map((p, i) => (i === index ? value : p)))
  }

  function getValidPaths() {
    return modelPaths.filter((p) => p.trim())
  }

  async function handleBenchmark() {
    const paths = getValidPaths()
    if (paths.length === 0 || !datasetPath.trim()) {
      toast.error('At least one model path and a dataset path are required')
      return
    }
    setRunningBenchmark(true)
    try {
      const res = await executeTool({
        toolName: 'evaluate_model.batch',
        args: { model_paths: paths, dataset_path: datasetPath },
      })
      setBenchmarkResult(res as Record<string, unknown>)
      toast.success('Benchmark complete')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Benchmark failed')
    } finally {
      setRunningBenchmark(false)
    }
  }

  async function handleCompare() {
    const paths = getValidPaths()
    if (paths.length < 2) {
      toast.error('At least two model paths are required for comparison')
      return
    }
    setRunningCompare(true)
    try {
      const res = await executeTool({
        toolName: 'evaluate_model.compare_models',
        args: { model_paths: paths },
      })
      setCompareResult(res as Record<string, unknown>)
      toast.success('Comparison complete')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Comparison failed')
    } finally {
      setRunningCompare(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="pt-6 space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Model Paths</label>
            {modelPaths.map((path, index) => (
              <div key={index} className="flex gap-2">
                <Input
                  placeholder={`/path/to/model-${index + 1}`}
                  value={path}
                  onChange={(e) => updateModel(index, e.target.value)}
                />
                {modelPaths.length > 1 && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeModel(index)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
              </div>
            ))}
            <Button variant="outline" size="sm" onClick={addModel}>
              <Plus className="h-3.5 w-3.5" />
              Add Model
            </Button>
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">Dataset / Benchmark Path</label>
            <Input
              placeholder="/path/to/benchmark-dataset.jsonl"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
            />
          </div>

          <div className="flex gap-2">
            <Button onClick={handleBenchmark} disabled={runningBenchmark}>
              {runningBenchmark ? 'Running...' : 'Run Benchmark'}
            </Button>
            <Button
              variant="outline"
              onClick={handleCompare}
              disabled={runningCompare}
            >
              {runningCompare ? 'Comparing...' : 'Compare Models'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {benchmarkResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Benchmark Results
              <ExportButton
                toolName="evaluate_model.export_results"
                args={{ model_paths: getValidPaths(), dataset_path: datasetPath }}
              />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <MetricsTable data={benchmarkResult} />
          </CardContent>
        </Card>
      )}

      {compareResult && (
        <Card>
          <CardHeader>
            <CardTitle>Model Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <MetricsTable data={compareResult} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
