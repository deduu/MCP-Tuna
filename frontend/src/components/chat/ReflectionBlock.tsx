import { CheckCircle2, XCircle } from 'lucide-react'
import type { ReflectionEvent } from '@/stores/chat'

interface ReflectionBlockProps {
  reflection: ReflectionEvent
}

export function ReflectionBlock({ reflection }: ReflectionBlockProps) {
  return (
    <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-border/50 bg-muted/20">
      {reflection.isReady ? (
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
      ) : (
        <XCircle className="h-3.5 w-3.5 text-amber-400 shrink-0 mt-0.5" />
      )}
      <div>
        <span className="text-xs font-medium">
          {reflection.isReady ? 'Ready to answer' : 'Needs more work'}
        </span>
        {reflection.explanation && (
          <p className="text-[11px] text-muted-foreground mt-0.5 leading-relaxed">
            {reflection.explanation}
          </p>
        )}
      </div>
    </div>
  )
}
