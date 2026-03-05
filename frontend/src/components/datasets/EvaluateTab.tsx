import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useEvalMetrics } from '@/api/hooks/useDatasets'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { BarChart3, Filter, Activity } from 'lucide-react'
import { toast } from 'sonner'
import { DatasetSelector } from './DatasetSelector'

export function EvaluateTab() {
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const { data: availableMetrics } = useEvalMetrics()

  const [selectedDataset, setSelectedDataset] = useState('')
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([])
  const [evalResult, setEvalResult] = useState<Record<string, unknown> | null>(null)
  const [statsResult, setStatsResult] = useState<Record<string, unknown> | null>(null)
  const [qualityThreshold, setQualityThreshold] = useState(0.7)

  function toggleMetric(metric: string) {
    setSelectedMetrics((prev) =>
      prev.includes(metric) ? prev.filter((m) => m !== metric) : [...prev, metric],
    )
  }

  async function handleEvaluate() {
    if (!selectedDataset || selectedMetrics.length === 0) return
    try {
      const result = await executeTool({
        toolName: 'evaluate.dataset',
        args: { dataset_path: selectedDataset, metrics: selectedMetrics },
      })
      setEvalResult(result as Record<string, unknown>)
      toast.success('Evaluation complete')
    } catch (err) {
      toast.error(`Evaluation failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleViewStats() {
    if (!selectedDataset) return
    try {
      const result = await executeTool({
        toolName: 'evaluate.statistics',
        args: { dataset_path: selectedDataset },
      })
      setStatsResult(result as Record<string, unknown>)
    } catch (err) {
      toast.error(`Stats failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleFilterByQuality() {
    if (!selectedDataset) return
    try {
      await executeTool({
        toolName: 'evaluate.filter_by_quality',
        args: { dataset_path: selectedDataset, threshold: qualityThreshold },
      })
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success(`Filtered dataset by quality >= ${qualityThreshold}`)
    } catch (err) {
      toast.error(`Filter failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <DatasetSelector
        value={selectedDataset}
        onChange={setSelectedDataset}
        label="Select dataset to evaluate"
      />

      {/* Metrics Selection & Evaluate */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <BarChart3 className="h-4 w-4" />
            Run Evaluation
          </CardTitle>
          <CardDescription>Select metrics and evaluate dataset quality</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm font-medium">Metrics</p>
            <div className="flex flex-wrap gap-2">
              {availableMetrics?.map((metric) => (
                <button
                  key={metric}
                  onClick={() => toggleMetric(metric)}
                  className="cursor-pointer"
                >
                  <Badge
                    variant={selectedMetrics.includes(metric) ? 'default' : 'outline'}
                  >
                    {metric}
                  </Badge>
                </button>
              ))}
              {!availableMetrics?.length && (
                <p className="text-xs text-muted-foreground">Loading metrics...</p>
              )}
            </div>
          </div>

          <Button
            onClick={handleEvaluate}
            disabled={isPending || !selectedDataset || selectedMetrics.length === 0}
          >
            Run Evaluation
          </Button>

          {evalResult && (
            <div className="space-y-2">
              <p className="text-sm font-medium">Results</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {Object.entries(evalResult).map(([key, value]) => {
                  if (key === 'success') return null
                  return (
                    <div
                      key={key}
                      className="rounded-lg border border-border p-3 text-center"
                    >
                      <p className="text-xs text-muted-foreground">{key}</p>
                      <p className="text-lg font-semibold">
                        {typeof value === 'number' ? value.toFixed(3) : String(value)}
                      </p>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Statistics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Activity className="h-4 w-4" />
            Statistics
          </CardTitle>
          <CardDescription>View detailed dataset statistics</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button
            variant="secondary"
            onClick={handleViewStats}
            disabled={isPending || !selectedDataset}
          >
            View Statistics
          </Button>

          {statsResult && (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {Object.entries(statsResult).map(([key, value]) => {
                if (key === 'success') return null
                return (
                  <div
                    key={key}
                    className="rounded-lg border border-border p-3 text-center"
                  >
                    <p className="text-xs text-muted-foreground">{key}</p>
                    <p className="text-sm font-semibold">
                      {typeof value === 'number' ? value.toFixed(3) : String(value)}
                    </p>
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Filter by Quality */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Filter className="h-4 w-4" />
            Filter by Quality
          </CardTitle>
          <CardDescription>Remove low-quality entries below a threshold</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <label className="text-xs text-muted-foreground mb-1 block">
                Quality threshold: {qualityThreshold}
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={qualityThreshold}
                onChange={(e) => setQualityThreshold(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <Input
              type="number"
              value={qualityThreshold}
              onChange={(e) => setQualityThreshold(Number(e.target.value))}
              min={0}
              max={1}
              step={0.05}
              className="w-20"
            />
          </div>
          <Button
            onClick={handleFilterByQuality}
            disabled={isPending || !selectedDataset}
          >
            Apply Filter
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}
