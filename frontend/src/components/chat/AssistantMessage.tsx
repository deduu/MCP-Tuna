import { Fish, Loader2 } from 'lucide-react'
import type { ChatMessage } from '@/stores/chat'
import { ThinkingBlock } from './ThinkingBlock'
import { ToolCallCard } from './ToolCallCard'
import { ReflectionBlock } from './ReflectionBlock'
import { TurnMetricsBadge } from './TurnMetricsBadge'
import { MarkdownContent } from './MarkdownContent'

interface AssistantMessageProps {
  message: ChatMessage
}

export function AssistantMessage({ message }: AssistantMessageProps) {
  const { content, thinking, toolCalls, reflections, metrics, isStreaming } = message

  // Determine which tool is currently running (started but no duration yet)
  const runningTools = new Set(
    toolCalls.filter((tc) => tc.durationMs === undefined).map((tc) => tc.tool),
  )

  const hasActivity = thinking.length > 0 || toolCalls.length > 0 || reflections.length > 0

  return (
    <div className="flex gap-3 max-w-3xl">
      <div className="shrink-0 mt-1">
        <div className="h-7 w-7 rounded-full bg-primary/10 flex items-center justify-center">
          <Fish className="h-4 w-4 text-primary" />
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-2">
        {/* Agent activity blocks (thinking, tools, reflection) */}
        {hasActivity && (
          <div className="space-y-1.5">
            {thinking.length > 0 && <ThinkingBlock thoughts={thinking} />}

            {toolCalls.map((tc, i) => (
              <ToolCallCard
                key={`${tc.tool}-${i}`}
                toolCall={tc}
                isRunning={runningTools.has(tc.tool)}
              />
            ))}

            {reflections.map((r, i) => (
              <ReflectionBlock key={i} reflection={r} />
            ))}
          </div>
        )}

        {/* Response content */}
        {(content || isStreaming) && (
          <div>
            {content && <MarkdownContent content={content} />}
            {isStreaming && !content && !hasActivity && (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
            {isStreaming && content && (
              <span className="inline-block w-1.5 h-4 bg-primary/70 animate-pulse ml-0.5 align-text-bottom" />
            )}
          </div>
        )}

        {/* Turn metrics (shown after completion) */}
        {!isStreaming && metrics.length > 0 && (
          <TurnMetricsBadge metrics={metrics} />
        )}
      </div>
    </div>
  )
}
