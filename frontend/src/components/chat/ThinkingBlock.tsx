import { useState } from 'react'
import { Brain, ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ThinkingBlockProps {
  thoughts: string[]
}

export function ThinkingBlock({ thoughts }: ThinkingBlockProps) {
  const [expanded, setExpanded] = useState(false)

  if (thoughts.length === 0) return null

  return (
    <button
      type="button"
      onClick={() => setExpanded(!expanded)}
      className={cn(
        'w-full text-left rounded-lg border border-border/50 transition-colors',
        'bg-muted/30 hover:bg-muted/50',
      )}
    >
      <div className="flex items-center gap-2 px-3 py-2">
        <Brain className="h-3.5 w-3.5 text-violet-400 shrink-0" />
        <span className="text-xs font-medium text-violet-400">Thinking</span>
        <span className="text-[10px] text-muted-foreground ml-auto">
          {thoughts.length} {thoughts.length === 1 ? 'step' : 'steps'}
        </span>
        {expanded ? (
          <ChevronDown className="h-3 w-3 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 text-muted-foreground" />
        )}
      </div>
      {expanded && (
        <div className="px-3 pb-3 space-y-2 border-t border-border/30 pt-2">
          {thoughts.map((thought, i) => (
            <div key={i} className="text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap">
              {thought}
            </div>
          ))}
        </div>
      )}
    </button>
  )
}
