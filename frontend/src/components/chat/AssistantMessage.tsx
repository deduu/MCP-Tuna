import { Fish, Loader2 } from 'lucide-react'
import type { ChatMessage } from '@/stores/chat'
import { useChatStore } from '@/stores/chat'
import { sendChatMessage } from '@/api/chat-client'
import { ThinkingBlock } from './ThinkingBlock'
import { ToolCallCard } from './ToolCallCard'
import { ReflectionBlock } from './ReflectionBlock'
import { TurnMetricsBadge } from './TurnMetricsBadge'
import { MarkdownContent } from './MarkdownContent'
import { ConfirmationCard } from './ConfirmationCard'

interface AssistantMessageProps {
  message: ChatMessage
}

export function AssistantMessage({ message }: AssistantMessageProps) {
  const { content, thinking, toolCalls, reflections, metrics, isStreaming, confirmation } = message
  const chatIsStreaming = useChatStore((s) => s.isStreaming)

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

        {/* Confirmation card */}
        {confirmation && (
          <ConfirmationCard
            confirmation={confirmation}
            onProceed={() => {
              useChatStore.getState().resolveConfirmation(message.id)
              sendChatMessage(`Confirmed: proceed with ${confirmation.tool}`)
            }}
            onCancel={() => {
              useChatStore.getState().resolveConfirmation(message.id)
              sendChatMessage(`Cancel: do not execute ${confirmation.tool}`)
            }}
            onModify={() => {
              useChatStore.getState().resolveConfirmation(message.id)
              // Focus is handled by the chat input component
            }}
            disabled={chatIsStreaming}
          />
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
