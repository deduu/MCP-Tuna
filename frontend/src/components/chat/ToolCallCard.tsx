import { useState } from 'react'
import { Wrench, ChevronDown, ChevronRight, Clock, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { NAMESPACE_MAP, getNamespaceFromToolName, getToolShortName } from '@/lib/tool-registry'
import type { ToolCallEvent } from '@/stores/chat'

interface ToolCallCardProps {
  toolCall: ToolCallEvent
  isRunning?: boolean
}

export function ToolCallCard({ toolCall, isRunning }: ToolCallCardProps) {
  const [expanded, setExpanded] = useState(false)

  const ns = getNamespaceFromToolName(toolCall.tool)
  const shortName = getToolShortName(toolCall.tool)
  const nsInfo = NAMESPACE_MAP[ns]
  const hasArgs = Object.keys(toolCall.arguments).length > 0

  return (
    <div
      className={cn(
        'rounded-lg border transition-colors',
        isRunning
          ? 'border-primary/30 bg-primary/5'
          : 'border-border/50 bg-muted/20',
      )}
    >
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left"
      >
        {isRunning ? (
          <Loader2 className="h-3.5 w-3.5 text-primary animate-spin shrink-0" />
        ) : (
          <Wrench className="h-3.5 w-3.5 shrink-0" style={{ color: nsInfo?.color }} />
        )}
        <span
          className="text-[10px] font-medium px-1.5 py-0.5 rounded"
          style={{
            color: nsInfo?.color,
            backgroundColor: `color-mix(in srgb, ${nsInfo?.color ?? '#888'} 15%, transparent)`,
          }}
        >
          {nsInfo?.label ?? ns}
        </span>
        <span className="text-xs font-medium">{shortName}</span>

        <div className="ml-auto flex items-center gap-2">
          {toolCall.durationMs !== undefined && (
            <span className="flex items-center gap-1 text-[10px] text-muted-foreground font-mono">
              <Clock className="h-2.5 w-2.5" />
              {toolCall.durationMs < 1000
                ? `${Math.round(toolCall.durationMs)}ms`
                : `${(toolCall.durationMs / 1000).toFixed(1)}s`}
            </span>
          )}
          {hasArgs && (
            expanded ? (
              <ChevronDown className="h-3 w-3 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
            )
          )}
        </div>
      </button>

      {expanded && hasArgs && (
        <div className="px-3 pb-3 border-t border-border/30 pt-2">
          <div className="text-[10px] text-muted-foreground font-medium mb-1">Arguments</div>
          <pre className="text-[11px] font-mono text-muted-foreground bg-background/50 rounded p-2 overflow-x-auto max-h-48">
            {JSON.stringify(toolCall.arguments, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
