import { Bot, Server, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { MarkdownContent } from './MarkdownContent'
import { MessageBlocks } from './MessageBlocks'
import { ReflectionBlock } from './ReflectionBlock'
import { ThinkingBlock } from './ThinkingBlock'
import { ToolCallCard } from './ToolCallCard'
import { CompareMetricsSummary, InlineMetricRow } from './CompareMetricsSummary'
import type { CompareMessage, CompareMetrics, CompareSession } from '@/stores/chatCompare'

interface ComparePaneProps {
  session: CompareSession
  baselineMetrics: CompareMetrics | null
  isBaseline: boolean
  onClear: () => void
  disabled: boolean
}

export function ComparePane({
  session,
  baselineMetrics,
  isBaseline,
  onClear,
  disabled,
}: ComparePaneProps) {
  const latestMetrics = getLatestAssistantMetrics(session.messages)

  return (
    <div className="min-h-0 rounded-xl border bg-card">
      <div className="flex items-center justify-between gap-3 border-b px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {session.target.kind === 'agent' ? (
              <Bot className="h-4 w-4 text-primary" />
            ) : (
              <Server className="h-4 w-4 text-amber-400" />
            )}
            <h3 className="truncate text-sm font-semibold">{session.target.label}</h3>
            {isBaseline && <Badge variant="success">Baseline</Badge>}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {session.target.kind === 'agent'
              ? session.target.model
              : session.target.deploymentLabel ?? session.target.deploymentId ?? 'No deployment selected'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge
            variant={
              session.status === 'streaming'
                ? 'warning'
                : session.status === 'error'
                  ? 'destructive'
                  : 'outline'
            }
          >
            {session.status === 'streaming'
              ? 'Running'
              : session.status === 'error'
                ? 'Error'
                : 'Ready'}
          </Badge>
          <Button variant="ghost" size="sm" onClick={onClear} disabled={disabled || session.messages.length === 0}>
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="border-b px-4 py-3">
        <CompareMetricsSummary metrics={latestMetrics} baselineMetrics={baselineMetrics} isBaseline={isBaseline} />
      </div>

      <div className="h-[calc(100%-9.5rem)] overflow-y-auto px-4 py-4">
        {session.messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center">
            <p className="max-w-sm text-sm text-muted-foreground">
              This pane will show the shared prompt history and responses for {session.target.label}.
            </p>
          </div>
        ) : (
          <div className="space-y-5">
            {session.messages.map((message) =>
              message.role === 'user' ? (
                <CompareUserMessage key={message.id} message={message} />
              ) : (
                <CompareAssistantMessage key={message.id} message={message} />
              ),
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function CompareUserMessage({ message }: { message: CompareMessage }) {
  return (
    <div className="space-y-2">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Prompt</div>
      {message.parts ? (
        <MessageBlocks blocks={message.parts} />
      ) : (
        <div className="whitespace-pre-wrap rounded-lg border border-border/70 bg-secondary/20 px-3 py-2 text-sm">
          {message.content}
        </div>
      )}
    </div>
  )
}

function CompareAssistantMessage({ message }: { message: CompareMessage }) {
  const runningTools = new Set(
    message.toolCalls.filter((toolCall) => toolCall.durationMs === undefined).map((toolCall) => toolCall.tool),
  )
  const hasActivity =
    message.thinking.length > 0 || message.toolCalls.length > 0 || message.reflections.length > 0

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-muted-foreground">
        <span>Response</span>
        {message.modelId && (
          <span className="truncate rounded bg-secondary/20 px-1.5 py-0.5 normal-case text-[10px]">
            {message.modelId}
          </span>
        )}
      </div>

      {hasActivity && (
        <div className="space-y-1.5">
          {message.thinking.length > 0 && <ThinkingBlock thoughts={message.thinking} />}
          {message.toolCalls.map((toolCall, index) => (
            <ToolCallCard
              key={`${toolCall.tool}-${index}`}
              toolCall={toolCall}
              isRunning={runningTools.has(toolCall.tool)}
            />
          ))}
          {message.reflections.map((reflection, index) => (
            <ReflectionBlock key={`${reflection.explanation}-${index}`} reflection={reflection} />
          ))}
        </div>
      )}

      <div
        className={cn(
          'rounded-lg border px-3 py-2',
          message.error ? 'border-destructive/50 bg-destructive/5' : 'border-border/70',
        )}
      >
        {message.content ? (
          <MarkdownContent content={message.content} />
        ) : (
          <p className="text-sm text-muted-foreground">
            {message.isStreaming ? 'Waiting for response...' : 'No content returned.'}
          </p>
        )}
        {message.isStreaming && (
          <span className="inline-block h-4 w-1.5 animate-pulse bg-primary/70 align-text-bottom" />
        )}
      </div>

      {message.metrics && <InlineMetricRow metrics={message.metrics} />}
    </div>
  )
}

function getLatestAssistantMetrics(messages: CompareMessage[]) {
  const lastAssistant = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant' && !message.isStreaming && message.metrics)
  return lastAssistant?.metrics ?? null
}
