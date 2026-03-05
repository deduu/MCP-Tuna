import { Gauge, Coins, Zap } from 'lucide-react'
import type { TurnMetrics } from '@/stores/chat'

interface TurnMetricsBadgeProps {
  metrics: TurnMetrics[]
}

export function TurnMetricsBadge({ metrics }: TurnMetricsBadgeProps) {
  if (metrics.length === 0) return null

  const lastMetrics = metrics[metrics.length - 1]
  const totalTokens = metrics.reduce((sum, m) => sum + (m.tokens ?? 0), 0)

  return (
    <div className="flex items-center gap-3 text-[10px] text-muted-foreground font-mono">
      {lastMetrics.confidence && (
        <span className="flex items-center gap-1" title="Confidence">
          <Gauge className="h-2.5 w-2.5" />
          {lastMetrics.confidence}
        </span>
      )}
      {totalTokens > 0 && (
        <span className="flex items-center gap-1" title="Total tokens">
          <Coins className="h-2.5 w-2.5" />
          {totalTokens.toLocaleString()}
        </span>
      )}
      {metrics.length > 1 && (
        <span className="flex items-center gap-1" title="Turns">
          <Zap className="h-2.5 w-2.5" />
          {metrics.length} turns
        </span>
      )}
    </div>
  )
}
