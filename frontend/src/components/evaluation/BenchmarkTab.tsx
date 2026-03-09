import { useState } from 'react'
import { Plus, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { toast } from 'sonner'
import { MetricsTable } from './MetricsTable'

export function BenchmarkTab() {
  const [modelPaths, setModelPaths] = useState<string[]>([''])
  const [datasetPath, setDatasetPath] = useState('')
  const [benchmarkResult, setBenchmarkResult] = useState<Record<string, unknown> | null>(null)
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null)
  const [exportPath, setExportPath] = useState('output/eval_results.jsonl')
  const { mutateAsync: executeTool } = useToolExecution()
  const [runningBenchmark, setRunningBenchmark] = useState(false)
  const [runningCompare, setRunningCompare] = useState(false)
  const [exporting, setExporting] = useState(false)

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

  async function loadTestData(path: string) {
    const loaded = await executeTool({
      toolName: 'dataset.load',
      args: { file_path: path },
    })
    const payload = loaded as Record<string, unknown>
    return Array.isArray(payload.data_points)
      ? (payload.data_points as Array<Record<string, unknown>>)
      : []
  }

  async function handleBenchmark() {
    const paths = getValidPaths()
    if (paths.length === 0 || !datasetPath.trim()) {
      toast.error('At least one model path and a dataset path are required')
      return
    }
    setRunningBenchmark(true)
    try {
      const testData = await loadTestData(datasetPath)
      const runs: Array<Record<string, unknown>> = []
      for (const modelPath of paths) {
        const res = await executeTool({
          toolName: 'evaluate_model.batch',
          args: { test_data: testData, model_path: modelPath },
        })
        runs.push({ model_path: modelPath, ...(res as Record<string, unknown>) })
      }
      setBenchmarkResult({
        success: true,
        count: runs.length,
        runs,
      })
      toast.success('Benchmark complete')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Benchmark failed')
    } finally {
      setRunningBenchmark(false)
    }
  }

  async function handleCompare() {
    const runs = (benchmarkResult?.runs as Array<Record<string, unknown>> | undefined) ?? []
    if (runs.length < 2) {
      toast.error('Run benchmark on at least two models first')
      return
    }
    setRunningCompare(true)
    try {
      const compare = runs.slice(0, 2).map((r) => ({
        model_path: r.model_path,
        summary: r.summary ?? {},
      }))
      setCompareResult({ success: true, compared: compare })
      toast.success('Comparison prepared')
    } finally {
      setRunningCompare(false)
    }
  }

  async function handleExport() {
    const runs = (benchmarkResult?.runs as Array<Record<string, unknown>> | undefined) ?? []
    const first = runs[0]
    if (!first || !Array.isArray(first.results)) {
      toast.error('No benchmark results available to export')
      return
    }
    setExporting(true)
    try {
      await executeTool({
        toolName: 'evaluate_model.export',
        args: { results: first.results, output_path: exportPath, format: 'jsonl' },
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
            <label className="text-sm font-medium">Dataset Path</label>
            <Input
              placeholder="/path/to/benchmark-dataset.jsonl"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">Export Path</label>
            <Input
              placeholder="output/eval_results.jsonl"
              value={exportPath}
              onChange={(e) => setExportPath(e.target.value)}
            />
          </div>

          <div className="flex gap-2">
            <Button onClick={handleBenchmark} disabled={runningBenchmark}>
              {runningBenchmark ? 'Running...' : 'Run Benchmark'}
            </Button>
            <Button
              variant="outline"
              onClick={handleCompare}
              disabled={runningCompare || !benchmarkResult}
            >
              {runningCompare ? 'Comparing...' : 'Compare Models'}
            </Button>
            <Button
              variant="outline"
              onClick={handleExport}
              disabled={exporting || !benchmarkResult}
            >
              {exporting ? 'Exporting...' : 'Export First Run'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {benchmarkResult && (
        <Card>
          <CardHeader>
            <CardTitle>Benchmark Results</CardTitle>
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
