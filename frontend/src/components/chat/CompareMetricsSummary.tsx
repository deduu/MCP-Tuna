import { Coins, Gauge, Timer, Wrench } from 'lucide-react'
import type { CompareMetrics } from '@/stores/chatCompare'

interface CompareMetricsSummaryProps {
  metrics: CompareMetrics | null
  baselineMetrics: CompareMetrics | null
  isBaseline: boolean
}

export function CompareMetricsSummary({
  metrics,
  baselineMetrics,
  isBaseline,
}: CompareMetricsSummaryProps) {
  if (!metrics) {
    return (
      <p className="text-xs text-muted-foreground">
        Run a compare prompt to capture latency, tokens, tool calls, and cost for this target.
      </p>
    )
  }

  return (
    <div className="space-y-2">
      <InlineMetricRow metrics={metrics} />
      {!isBaseline && baselineMetrics && (
        <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
          {renderDelta('Latency', metrics.latency_ms, baselineMetrics.latency_ms, 'ms')}
          {renderDelta('TPS', metrics.output_tokens_per_second, baselineMetrics.output_tokens_per_second, 't/s', true)}
          {renderDelta('Tokens', metrics.total_tokens, baselineMetrics.total_tokens)}
          {renderDelta('Cost', metrics.estimated_cost_usd, baselineMetrics.estimated_cost_usd, 'USD')}
        </div>
      )}
    </div>
  )
}

export function InlineMetricRow({ metrics }: { metrics: CompareMetrics }) {
  return (
    <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono text-muted-foreground">
      {metrics.latency_ms != null && (
        <span className="flex items-center gap-1" title="Total latency">
          <Timer className="h-3 w-3" />
          {formatNumber(metrics.latency_ms)}ms
        </span>
      )}
      {metrics.ttft_ms != null && (
        <span className="flex items-center gap-1" title="Time to first token">
          <Gauge className="h-3 w-3" />
          {formatNumber(metrics.ttft_ms)}ms TTFT
        </span>
      )}
      {metrics.total_tokens != null && (
        <span className="flex items-center gap-1" title="Total tokens">
          <Coins className="h-3 w-3" />
          {formatNumber(metrics.total_tokens)}
        </span>
      )}
      {metrics.output_tokens_per_second != null && (
        <span className="flex items-center gap-1" title="Output tokens per second">
          <Gauge className="h-3 w-3" />
          {formatNumber(metrics.output_tokens_per_second)} t/s
        </span>
      )}
      {metrics.tool_call_count != null && metrics.tool_call_count > 0 && (
        <span className="flex items-center gap-1" title="Tool calls">
          <Wrench className="h-3 w-3" />
          {metrics.tool_call_count} calls
        </span>
      )}
      {metrics.estimated_cost_usd != null && (
        <span className="flex items-center gap-1" title="Estimated cost">
          <Coins className="h-3 w-3" />
          ${metrics.estimated_cost_usd.toFixed(4)}
        </span>
      )}
    </div>
  )
}

function renderDelta(
  label: string,
  value: number | null | undefined,
  baseline: number | null | undefined,
  suffix = '',
  invertDirection = false,
) {
  if (value == null || baseline == null) {
    return null
  }
  const delta = value - baseline
  if (delta === 0) {
    return <span key={label}>{label}: even</span>
  }
  const improved = invertDirection ? delta > 0 : delta < 0
  const sign = delta > 0 ? '+' : ''
  return (
    <span key={label} className={improved ? 'text-emerald-400' : 'text-amber-400'}>
      {label}: {sign}
      {formatNumber(delta)}
      {suffix ? ` ${suffix}` : ''}
    </span>
  )
}

function formatNumber(value: number) {
  if (!Number.isFinite(value)) {
    return '0'
  }
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(1)
}
