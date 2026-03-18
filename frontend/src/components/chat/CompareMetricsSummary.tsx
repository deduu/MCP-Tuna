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
      {isBaseline ? (
        <p className="text-[11px] text-muted-foreground">Baseline reference for compare deltas.</p>
      ) : baselineMetrics ? (
        <div className="space-y-1">
          <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Vs baseline</p>
          <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
            {renderDelta('Latency', metrics.latency_ms, baselineMetrics.latency_ms, 'ms')}
            {renderDelta(
              'Speed',
              metrics.output_tokens_per_second,
              baselineMetrics.output_tokens_per_second,
              't/s',
              true,
            )}
            {renderDelta('Tokens', metrics.total_tokens, baselineMetrics.total_tokens)}
            {renderDelta('Cost', metrics.estimated_cost_usd, baselineMetrics.estimated_cost_usd, 'USD')}
          </div>
        </div>
      ) : null}
    </div>
  )
}

export function InlineMetricRow({ metrics }: { metrics: CompareMetrics }) {
  const items = [
    metrics.latency_ms != null ? ['Latency', `${formatNumber(metrics.latency_ms)} ms`] : null,
    metrics.ttft_ms != null ? ['TTFT', `${formatNumber(metrics.ttft_ms)} ms`] : null,
    metrics.total_tokens != null ? ['Tokens', formatNumber(metrics.total_tokens)] : null,
    metrics.output_tokens_per_second != null
      ? ['Speed', `${formatNumber(metrics.output_tokens_per_second)} t/s`]
      : null,
    metrics.tool_call_count != null && metrics.tool_call_count > 0
      ? ['Tools', `${metrics.tool_call_count} calls`]
      : null,
    metrics.estimated_cost_usd != null ? ['Cost', `$${metrics.estimated_cost_usd.toFixed(4)}`] : null,
  ].filter((item): item is [string, string] => Boolean(item))

  return (
    <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
      {items.map(([label, value]) => (
        <span
          key={label}
          className="rounded-full border border-border/60 bg-secondary/15 px-2.5 py-1 font-mono"
          title={label}
        >
          <span className="mr-1 text-[10px] uppercase tracking-wide text-muted-foreground/80">{label}</span>
          <span className="text-foreground/90">{value}</span>
        </span>
      ))}
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
    <span
      key={label}
      className={`rounded-full border px-2.5 py-1 ${
        improved
          ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
          : 'border-amber-500/30 bg-amber-500/10 text-amber-300'
      }`}
    >
      {label} {sign}
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
