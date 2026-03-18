import type { CompareJudgement } from '@/stores/chatCompare'
import { Badge } from '@/components/ui/badge'

interface CompareJudgementCardProps {
  judgement: CompareJudgement
  targetLabel: string
}

export function CompareJudgementCard({
  judgement,
  targetLabel,
}: CompareJudgementCardProps) {
  const winnerLabel =
    judgement.winner === 'baseline'
      ? 'Baseline wins'
      : judgement.winner === 'target'
        ? `${targetLabel} wins`
        : 'Tie'

  const winnerVariant =
    judgement.winner === 'baseline'
      ? 'secondary'
      : judgement.winner === 'target'
        ? 'success'
        : 'outline'

  return (
    <div className="space-y-2 rounded-lg border border-border/70 bg-secondary/20 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={winnerVariant}>{winnerLabel}</Badge>
        {judgement.confidence != null && (
          <span className="text-xs text-muted-foreground">
            Confidence {judgement.confidence.toFixed(2)}
          </span>
        )}
        <span className="text-[10px] text-muted-foreground">{judgement.toolName}</span>
      </div>
      {judgement.rationale && (
        <p className="text-xs text-muted-foreground whitespace-pre-wrap">{judgement.rationale}</p>
      )}
    </div>
  )
}
