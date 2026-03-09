import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useDatasets, useEvalConfig, useEvalMetrics, type EvalConfig } from '@/api/hooks/useDatasets'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { buildDatasetOutputPath } from '@/lib/dataset-output'
import { Badge } from '@/components/ui/badge'
import { BarChart3, Filter, Activity, Loader2, Settings2, Files } from 'lucide-react'
import { toast } from 'sonner'
import { DatasetSelector } from './DatasetSelector'

const METRIC_DESCRIPTIONS: Record<string, string> = {
  complexity: 'Vocabulary richness, structure, and semantic density.',
  ifd: 'Instruction-following difficulty based on specificity and alignment.',
  quality: 'LLM-judged response quality.',
}

interface MetricStatsSummary {
  min: number
  max: number
  mean: number
  stdev: number
}

interface StatsResult {
  total_data_points: number
  statistics: Record<string, MetricStatsSummary>
}

function formatDatasetName(filePath: string): string {
  return filePath.split(/[\\/]/).pop() ?? filePath
}

function formatNumber(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '0.000'
}

function isMetricStatsSummary(value: unknown): value is MetricStatsSummary {
  if (!value || typeof value !== 'object') {
    return false
  }

  const record = value as Record<string, unknown>
  return ['min', 'max', 'mean', 'stdev'].every((key) => typeof record[key] === 'number')
}

export function EvaluateTab() {
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const { data: datasets = [] } = useDatasets()
  const {
    data: availableMetrics,
    isLoading: metricsLoading,
    error: metricsError,
  } = useEvalMetrics()
  const {
    data: evalConfig,
    isLoading: configLoading,
    error: configError,
  } = useEvalConfig()

  const [selectedDataset, setSelectedDataset] = useState('')
  const [multiDatasetMode, setMultiDatasetMode] = useState(false)
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([])
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([])
  const [evalResult, setEvalResult] = useState<Record<string, unknown> | null>(null)
  const [statsResult, setStatsResult] = useState<StatsResult | null>(null)
  const [qualityThreshold, setQualityThreshold] = useState(0.7)
  const [metricsWaitSeconds, setMetricsWaitSeconds] = useState(0)
  const [activeAction, setActiveAction] = useState<
    'evaluate' | 'stats' | 'filter' | 'config' | null
  >(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const [configDraft, setConfigDraft] = useState<EvalConfig | null>(null)

  useEffect(() => {
    if (!metricsLoading) {
      setMetricsWaitSeconds(0)
      return
    }

    const startedAt = Date.now()
    const timer = window.setInterval(() => {
      setMetricsWaitSeconds(Math.max(1, Math.floor((Date.now() - startedAt) / 1000)))
    }, 1000)

    return () => {
      window.clearInterval(timer)
    }
  }, [metricsLoading])

  useEffect(() => {
    if (!evalConfig) {
      return
    }

    setConfigDraft({
      ...evalConfig,
      weights: { ...evalConfig.weights },
    })
    setQualityThreshold(evalConfig.threshold)
  }, [evalConfig])

  function buildFilteredOutputPath(filePaths: string[]): string {
    const source = filePaths.length === 1 ? filePaths[0] : `combined_${filePaths.length}_datasets`
    return buildDatasetOutputPath(source, 'quality_filtered')
  }

  function getSelectedFilePaths(): string[] {
    return multiDatasetMode
      ? selectedDatasets
      : (selectedDataset ? [selectedDataset] : [])
  }

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

  async function loadSelectedDataPoints(filePaths: string[]) {
    const loaded = await Promise.all(filePaths.map((filePath) => loadDataPoints(filePath)))
    return loaded.flat()
  }

  function toggleMetric(metric: string) {
    setSelectedMetrics((prev) =>
      prev.includes(metric) ? prev.filter((m) => m !== metric) : [...prev, metric],
    )
  }

  function toggleDataset(filePath: string) {
    setSelectedDatasets((prev) =>
      prev.includes(filePath) ? prev.filter((path) => path !== filePath) : [...prev, filePath],
    )
  }

  function handleMultiDatasetToggle(enabled: boolean) {
    setMultiDatasetMode(enabled)
    if (enabled) {
      setSelectedDatasets((prev) => {
        if (!selectedDataset || prev.includes(selectedDataset)) {
          return prev
        }
        return [...prev, selectedDataset]
      })
      return
    }

    if (!selectedDataset && selectedDatasets.length > 0) {
      setSelectedDataset(selectedDatasets[0])
    }
  }

  function buildMetricSummary(metrics: string[]): string {
    return metrics.join(', ')
  }

  function buildSelectionSummary(filePaths: string[]): string {
    if (filePaths.length === 1) {
      return formatDatasetName(filePaths[0])
    }
    return `${filePaths.length} datasets combined`
  }

  function getMetricWeightKeys(): string[] {
    const keys = new Set([
      ...(availableMetrics ?? []),
      ...Object.keys(configDraft?.weights ?? {}),
    ])
    return Array.from(keys)
  }

  function resetConfigDraft() {
    if (!evalConfig) {
      return
    }

    setConfigDraft({
      ...evalConfig,
      weights: { ...evalConfig.weights },
    })
  }

  function updateWeight(metric: string, value: string) {
    setConfigDraft((prev) => {
      if (!prev) {
        return prev
      }

      return {
        ...prev,
        weights: {
          ...prev.weights,
          [metric]: Number(value),
        },
      }
    })
  }

  async function handleSaveConfig() {
    if (!configDraft) {
      return
    }

    setActiveAction('config')
    setActionMessage('Saving evaluator configuration for this gateway session...')
    try {
      const result = await executeTool({
        toolName: 'evaluate.update_config',
        args: {
          language: configDraft.language,
          threshold: configDraft.threshold,
          weights: configDraft.weights,
        },
      })
      const payload = result as { config?: EvalConfig }
      if (payload.config) {
        setConfigDraft({
          ...payload.config,
          weights: { ...payload.config.weights },
        })
        setQualityThreshold(payload.config.threshold)
      }
      queryClient.invalidateQueries({ queryKey: ['evaluate', 'config'] })
      toast.success('Evaluator config updated')
    } catch (err) {
      toast.error(`Config update failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setActiveAction(null)
      setActionMessage(null)
    }
  }

  async function handleEvaluate() {
    const filePaths = getSelectedFilePaths()
    if (filePaths.length === 0 || selectedMetrics.length === 0) return
    setActiveAction('evaluate')
    setActionMessage('Loading dataset for evaluation...')
    try {
      const dataPoints = await loadSelectedDataPoints(filePaths)
      if (dataPoints.length === 0) {
        toast.error('Dataset selection has no rows to evaluate')
        return
      }
      setActionMessage(
        `Evaluating ${dataPoints.length} row(s) from ${buildSelectionSummary(filePaths)} with ${buildMetricSummary(selectedMetrics)}.${selectedMetrics.includes('quality') ? ' Quality uses the LLM and may take longer.' : ''}`,
      )
      const result = await executeTool({
        toolName: 'evaluate.dataset',
        args: { data_points: dataPoints, metrics: selectedMetrics },
      })
      const payload = result as Record<string, unknown>
      const scored = Array.isArray(payload.data_points)
        ? (payload.data_points as Array<Record<string, unknown>>)
        : []
      const weighted = scored
        .map((r) => Number(r.weighted_score ?? 0))
        .filter((v) => Number.isFinite(v))
      const avgWeighted = weighted.length
        ? weighted.reduce((a, b) => a + b, 0) / weighted.length
        : 0
      setEvalResult({
        success: payload.success ?? true,
        count: payload.count ?? scored.length,
        avg_weighted_score: Number(avgWeighted.toFixed(4)),
      })
      toast.success('Evaluation complete')
    } catch (err) {
      toast.error(`Evaluation failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setActiveAction(null)
      setActionMessage(null)
    }
  }

  async function handleViewStats() {
    const filePaths = getSelectedFilePaths()
    if (filePaths.length === 0) return
    setActiveAction('stats')
    setActionMessage('Loading dataset statistics...')
    try {
      const dataPoints = await loadSelectedDataPoints(filePaths)
      setActionMessage(
        `Computing statistics for ${dataPoints.length} row(s) from ${buildSelectionSummary(filePaths)}.${selectedMetrics.length > 0 ? ` Using ${buildMetricSummary(selectedMetrics)}.` : ''}`,
      )
      const result = await executeTool({
        toolName: 'evaluate.statistics',
        args: {
          data_points: dataPoints,
          ...(selectedMetrics.length > 0 ? { metrics: selectedMetrics } : {}),
        },
      })
      const payload = result as Record<string, unknown>
      const rawStats = (payload.statistics as Record<string, unknown>) ?? {}
      const statistics = Object.fromEntries(
        Object.entries(rawStats).filter(([, value]) => isMetricStatsSummary(value)),
      ) as Record<string, MetricStatsSummary>
      setStatsResult({
        total_data_points: Number(payload.total_data_points ?? dataPoints.length),
        statistics,
      })
    } catch (err) {
      toast.error(`Stats failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setActiveAction(null)
      setActionMessage(null)
    }
  }

  async function handleFilterByQuality() {
    const filePaths = getSelectedFilePaths()
    if (filePaths.length === 0) return
    setActiveAction('filter')
    setActionMessage('Loading dataset for quality filtering...')
    try {
      const dataPoints = await loadSelectedDataPoints(filePaths)
      setActionMessage(
        `Filtering ${dataPoints.length} row(s) from ${buildSelectionSummary(filePaths)} at threshold ${qualityThreshold}.${selectedMetrics.length > 0 ? ` Using ${buildMetricSummary(selectedMetrics)}.` : ''}${selectedMetrics.includes('quality') ? ' Quality uses the LLM and may take longer.' : ''}`,
      )
      const filtered = await executeTool({
        toolName: 'evaluate.filter_by_quality',
        args: {
          data_points: dataPoints,
          threshold: qualityThreshold,
          ...(selectedMetrics.length > 0 ? { metrics: selectedMetrics } : {}),
        },
      })
      const payload = filtered as Record<string, unknown>
      const points = Array.isArray(payload.data_points)
        ? (payload.data_points as Array<Record<string, unknown>>)
        : []
      await executeTool({
        toolName: 'dataset.save',
        args: {
          data_points: points,
          output_path: buildFilteredOutputPath(filePaths),
          format: 'jsonl',
        },
      })
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success(`Filtered dataset by quality >= ${qualityThreshold}`)
    } catch (err) {
      toast.error(`Filter failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setActiveAction(null)
      setActionMessage(null)
    }
  }

  const selectedFilePaths = getSelectedFilePaths()
  const metricWeightKeys = getMetricWeightKeys()

  return (
    <div className="max-w-4xl space-y-6">
      <div className="space-y-3">
        <label className="flex cursor-pointer items-center gap-2 text-sm font-medium">
          <input
            type="checkbox"
            checked={multiDatasetMode}
            onChange={(e) => handleMultiDatasetToggle(e.target.checked)}
          />
          Evaluate multiple datasets together
        </label>

        {multiDatasetMode ? (
          <div className="space-y-2 rounded-lg border border-border/60 p-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Files className="h-4 w-4" />
              Select datasets to combine
            </div>
            <div className="max-h-56 space-y-2 overflow-auto">
              {datasets.map((dataset) => (
                <label
                  key={dataset.file_path}
                  className="flex cursor-pointer items-center justify-between gap-3 rounded-md border border-border/50 px-3 py-2 text-sm"
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <input
                      type="checkbox"
                      checked={selectedDatasets.includes(dataset.file_path)}
                      onChange={() => toggleDataset(dataset.file_path)}
                    />
                    <span className="truncate">{formatDatasetName(dataset.file_path)}</span>
                  </div>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {dataset.row_count.toLocaleString()} rows
                  </span>
                </label>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              {selectedFilePaths.length
                ? `Selected ${selectedFilePaths.length} dataset(s). Their rows will be concatenated before evaluation.`
                : 'Select one or more datasets to evaluate together.'}
            </p>
          </div>
        ) : (
          <DatasetSelector
            value={selectedDataset}
            onChange={setSelectedDataset}
            label="Select dataset to evaluate"
          />
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Settings2 className="h-4 w-4" />
            Evaluation Config
          </CardTitle>
          <CardDescription>
            Adjust weights, threshold, and language for this gateway session.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {configLoading && (
            <div className="rounded-md border border-border/60 bg-secondary/20 px-3 py-2 text-xs text-muted-foreground">
              Loading evaluator configuration...
            </div>
          )}
          {!configLoading && configError && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {configError.message}
            </div>
          )}
          {configDraft && (
            <>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Language</label>
                  <Input
                    value={configDraft.language}
                    onChange={(e) =>
                      setConfigDraft((prev) =>
                        prev
                          ? {
                              ...prev,
                              language: e.target.value,
                            }
                          : prev,
                      )
                    }
                    placeholder="en"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Default quality threshold</label>
                  <Input
                    type="number"
                    value={configDraft.threshold}
                    onChange={(e) =>
                      setConfigDraft((prev) =>
                        prev
                          ? {
                              ...prev,
                              threshold: Number(e.target.value),
                            }
                          : prev,
                      )
                    }
                    min={0}
                    max={1}
                    step={0.05}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <p className="text-sm font-medium">Metric weights</p>
                <div className="grid gap-3 sm:grid-cols-3">
                  {metricWeightKeys.map((metric) => (
                    <div key={`${metric}-weight`} className="space-y-1.5">
                      <label className="text-sm font-medium">{metric}</label>
                      <Input
                        type="number"
                        value={configDraft.weights[metric] ?? 0}
                        onChange={(e) => updateWeight(metric, e.target.value)}
                        step={0.05}
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                <Button
                  variant="secondary"
                  onClick={handleSaveConfig}
                  disabled={isPending || !configDraft}
                >
                  {activeAction === 'config' && <Loader2 className="h-4 w-4 animate-spin" />}
                  {activeAction === 'config' ? 'Saving...' : 'Save Config'}
                </Button>
                <Button variant="secondary" onClick={resetConfigDraft} disabled={isPending || !evalConfig}>
                  Reset Form
                </Button>
              </div>
              {activeAction === 'config' && actionMessage && (
                <div className="rounded-md border border-border/60 bg-secondary/20 px-3 py-2 text-xs text-muted-foreground">
                  {actionMessage}
                </div>
              )}
              <p className="text-xs text-muted-foreground">
                This updates the current gateway session only. Adding a brand-new metric still
                requires backend code that implements a new metric class.
              </p>
            </>
          )}
        </CardContent>
      </Card>

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
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2">
                {availableMetrics?.map((metric) => (
                  <button
                    key={metric}
                    onClick={() => toggleMetric(metric)}
                    className="cursor-pointer"
                    disabled={!!metricsError}
                  >
                    <Badge variant={selectedMetrics.includes(metric) ? 'default' : 'outline'}>
                      {metric}
                    </Badge>
                  </button>
                ))}
                {metricsLoading && (
                  <div className="flex items-start gap-2 rounded-md border border-border/60 bg-secondary/20 px-3 py-2 text-xs text-muted-foreground">
                    <Loader2 className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin" />
                    <div className="space-y-1">
                      <p>
                        Preparing evaluation metrics
                        {metricsWaitSeconds > 0 ? ` (${metricsWaitSeconds}s)` : ''}.
                      </p>
                      <p>
                        First load can take longer while the gateway warms evaluation components.
                      </p>
                    </div>
                  </div>
                )}
                {!metricsLoading && !!metricsError && (
                  <div className="space-y-1">
                    <p className="text-xs text-destructive">{metricsError.message}</p>
                    <p className="text-xs text-muted-foreground">
                      Evaluation tools are unavailable on the current gateway.
                    </p>
                  </div>
                )}
                {!metricsLoading && !metricsError && !availableMetrics?.length && (
                  <p className="text-xs text-muted-foreground">No metrics are registered.</p>
                )}
              </div>
              {!!availableMetrics?.length && (
                <div className="grid gap-2 sm:grid-cols-3">
                  {availableMetrics.map((metric) => (
                    <div key={`${metric}-desc`} className="rounded-md border border-border/60 p-3">
                      <p className="text-sm font-medium">{metric}</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {METRIC_DESCRIPTIONS[metric] ?? 'Registered evaluation metric.'}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <Button
            onClick={handleEvaluate}
            disabled={
              isPending ||
              !!metricsError ||
              selectedFilePaths.length === 0 ||
              selectedMetrics.length === 0
            }
          >
            {activeAction === 'evaluate' && <Loader2 className="h-4 w-4 animate-spin" />}
            {activeAction === 'evaluate' ? 'Running...' : 'Run Evaluation'}
          </Button>
          {activeAction === 'evaluate' && actionMessage && (
            <div className="rounded-md border border-border/60 bg-secondary/20 px-3 py-2 text-xs text-muted-foreground">
              {actionMessage}
            </div>
          )}

          {evalResult && (
            <div className="space-y-2">
              <p className="text-sm font-medium">Results</p>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {Object.entries(evalResult).map(([key, value]) => {
                  if (key === 'success') return null
                  return (
                    <div key={key} className="rounded-lg border border-border p-3 text-center">
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
            disabled={isPending || !!metricsError || selectedFilePaths.length === 0}
          >
            {activeAction === 'stats' && <Loader2 className="h-4 w-4 animate-spin" />}
            {activeAction === 'stats' ? 'Computing...' : 'View Statistics'}
          </Button>
          {activeAction === 'stats' && actionMessage && (
            <div className="rounded-md border border-border/60 bg-secondary/20 px-3 py-2 text-xs text-muted-foreground">
              {actionMessage}
            </div>
          )}

          {statsResult && (
            <div className="space-y-3">
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg border border-border p-3 text-center">
                  <p className="text-xs text-muted-foreground">total_data_points</p>
                  <p className="text-lg font-semibold">
                    {statsResult.total_data_points.toLocaleString()}
                  </p>
                </div>
                <div className="rounded-lg border border-border p-3 text-center">
                  <p className="text-xs text-muted-foreground">metrics_reported</p>
                  <p className="text-lg font-semibold">
                    {Object.keys(statsResult.statistics).length}
                  </p>
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                {Object.entries(statsResult.statistics).map(([metric, summary]) => (
                  <div key={metric} className="rounded-lg border border-border p-3">
                    <p className="text-sm font-medium">{metric}</p>
                    <div className="mt-3 grid grid-cols-2 gap-2">
                      <div className="rounded-md bg-secondary/20 p-2 text-center">
                        <p className="text-[11px] text-muted-foreground">mean</p>
                        <p className="text-sm font-semibold">{formatNumber(summary.mean)}</p>
                      </div>
                      <div className="rounded-md bg-secondary/20 p-2 text-center">
                        <p className="text-[11px] text-muted-foreground">stdev</p>
                        <p className="text-sm font-semibold">{formatNumber(summary.stdev)}</p>
                      </div>
                      <div className="rounded-md bg-secondary/20 p-2 text-center">
                        <p className="text-[11px] text-muted-foreground">min</p>
                        <p className="text-sm font-semibold">{formatNumber(summary.min)}</p>
                      </div>
                      <div className="rounded-md bg-secondary/20 p-2 text-center">
                        <p className="text-[11px] text-muted-foreground">max</p>
                        <p className="text-sm font-semibold">{formatNumber(summary.max)}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

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
              <label className="mb-1 block text-xs text-muted-foreground">
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
            disabled={isPending || !!metricsError || selectedFilePaths.length === 0}
          >
            {activeAction === 'filter' && <Loader2 className="h-4 w-4 animate-spin" />}
            {activeAction === 'filter' ? 'Filtering...' : 'Apply Filter'}
          </Button>
          {activeAction === 'filter' && actionMessage && (
            <div className="rounded-md border border-border/60 bg-secondary/20 px-3 py-2 text-xs text-muted-foreground">
              {actionMessage}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
