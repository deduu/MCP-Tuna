import { useMemo, useState } from 'react'
import { Bot, Scale, Server, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { judgeAgainstBaseline } from '@/lib/compare-judging'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { MarkdownContent } from './MarkdownContent'
import { CompareJudgementCard } from './CompareJudgementCard'
import { MessageBlocks } from './MessageBlocks'
import { ReflectionBlock } from './ReflectionBlock'
import { ThinkingBlock } from './ThinkingBlock'
import { ToolCallCard } from './ToolCallCard'
import { CompareMetricsSummary, InlineMetricRow } from './CompareMetricsSummary'
import { useChatCompareStore, type CompareMessage, type CompareMetrics, type CompareSession } from '@/stores/chatCompare'

interface ComparePaneProps {
  session: CompareSession
  baselineSession: CompareSession | null
  baselineMetrics: CompareMetrics | null
  isBaseline: boolean
  onClear: () => void
  disabled: boolean
}

export function ComparePane({
  session,
  baselineSession,
  baselineMetrics,
  isBaseline,
  onClear,
  disabled,
}: ComparePaneProps) {
  const setMessageJudgement = useChatCompareStore((state) => state.setMessageJudgement)
  const [judgingIds, setJudgingIds] = useState<Record<string, boolean>>({})
  const latestMetrics = getLatestAssistantMetrics(session.messages)
  const comparableAssistantIds = useMemo(
    () => buildComparableAssistantIds(session.messages, baselineSession?.messages ?? []),
    [baselineSession?.messages, session.messages],
  )

  async function handleJudgeMessage(message: CompareMessage) {
    if (isBaseline || !baselineSession) {
      return
    }
    const comparable = comparableAssistantIds[message.id]
    if (!comparable) {
      return
    }

    setJudgingIds((current) => ({ ...current, [message.id]: true }))
    try {
      const judgement = await judgeAgainstBaseline({
        promptText: comparable.userMessage.content,
        promptParts: comparable.userMessage.parts,
        baselineResponse: comparable.baselineAssistant.content,
        targetResponse: message.content,
      })
      setMessageJudgement(session.target.id, message.id, judgement)
    } catch (error) {
      setMessageJudgement(session.target.id, message.id, {
        winner: 'tie',
        confidence: null,
        rationale: error instanceof Error ? error.message : 'Judging failed',
        toolName: 'judge',
        judgedAt: new Date().toISOString(),
      })
    } finally {
      setJudgingIds((current) => {
        const next = { ...current }
        delete next[message.id]
        return next
      })
    }
  }

  return (
    <div className="flex min-h-[420px] flex-col overflow-hidden rounded-2xl border bg-card">
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

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {session.messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center">
            <p className="max-w-sm text-sm text-muted-foreground">
              This pane will show the shared prompt history and responses for {session.target.label}.
            </p>
          </div>
        ) : (
          <div className="space-y-6">
            {session.messages.map((message) =>
              message.role === 'user' ? (
                <CompareUserMessage key={message.id} message={message} />
              ) : (
                <CompareAssistantMessage
                  key={message.id}
                  message={message}
                  targetLabel={session.target.label}
                  isBaseline={isBaseline}
                  canJudge={Boolean(comparableAssistantIds[message.id])}
                  isJudging={Boolean(judgingIds[message.id])}
                  onJudge={() => void handleJudgeMessage(message)}
                />
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
        <div className="whitespace-pre-wrap rounded-xl border border-border/70 bg-secondary/20 px-4 py-3 text-sm leading-6">
          {message.content}
        </div>
      )}
    </div>
  )
}

function CompareAssistantMessage({
  message,
  targetLabel,
  isBaseline,
  canJudge,
  isJudging,
  onJudge,
}: {
  message: CompareMessage
  targetLabel: string
  isBaseline: boolean
  canJudge: boolean
  isJudging: boolean
  onJudge: () => void
}) {
  const runningTools = new Set(
    message.toolCalls.filter((toolCall) => toolCall.durationMs === undefined).map((toolCall) => toolCall.tool),
  )
  const hasActivity =
    message.thinking.length > 0 || message.toolCalls.length > 0 || message.reflections.length > 0

  return (
    <div className="space-y-3">
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
          'min-h-[120px] rounded-xl border px-4 py-3',
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
      {!isBaseline && !message.isStreaming && !message.error && message.content && canJudge && (
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" size="sm" onClick={onJudge} disabled={isJudging}>
            <Scale className="h-4 w-4" />
            {isJudging ? 'Judging...' : 'Judge Vs Baseline'}
          </Button>
          <span className="text-[11px] text-muted-foreground">
            Uses text or multimodal judge automatically based on the original prompt.
          </span>
        </div>
      )}
      {message.judgement && (
        <CompareJudgementCard judgement={message.judgement} targetLabel={targetLabel} />
      )}
    </div>
  )
}

function getLatestAssistantMetrics(messages: CompareMessage[]) {
  const lastAssistant = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant' && !message.isStreaming && message.metrics)
  return lastAssistant?.metrics ?? null
}

function buildComparableAssistantIds(
  targetMessages: CompareMessage[],
  baselineMessages: CompareMessage[],
) {
  const targetRounds = buildRounds(targetMessages)
  const baselineRounds = buildRounds(baselineMessages)
  const comparable: Record<string, { userMessage: CompareMessage; baselineAssistant: CompareMessage }> = {}

  for (let index = 0; index < targetRounds.length; index += 1) {
    const targetRound = targetRounds[index]
    const baselineRound = baselineRounds[index]
    if (!targetRound?.assistant || !baselineRound?.assistant) {
      continue
    }
    comparable[targetRound.assistant.id] = {
      userMessage: targetRound.user,
      baselineAssistant: baselineRound.assistant,
    }
  }
  return comparable
}

function buildRounds(messages: CompareMessage[]) {
  const users = messages.filter((message) => message.role === 'user')
  const assistants = messages.filter((message) => message.role === 'assistant')
  return users.map((user, index) => ({
    user,
    assistant: assistants[index],
  }))
}
