import { mcpCall } from '@/api/client'
import { streamDeploymentTextChat } from '@/api/deployment-chat-stream'
import {
  sanitizeChatContentForRequest,
  type ChatContentBlock,
} from '@/lib/chat-content'
import type { CompareMetrics, CompareTargetConfig } from '@/stores/chatCompare'

const CHAT_URL = '/v1/chat/completions'

interface SerializedMessage {
  role: 'user' | 'assistant'
  content: string | Array<Record<string, unknown>>
}

interface AgentCompareCallbacks {
  onToken: (token: string) => void
  onThinking: (content: string) => void
  onToolStart: (tool: string, args: Record<string, unknown>) => void
  onToolEnd: (tool: string, durationMs: number) => void
  onReflection: (isReady: boolean, explanation: string) => void
  onMetrics: (metrics: CompareMetrics) => void
  onComplete: (payload: { metrics?: CompareMetrics | null; modelId?: string | null }) => void
  onError: (message: string) => void
}

interface DeploymentCompareCallbacks {
  onToken: (token: string) => void
  onComplete: (payload: {
    response: string
    metrics?: CompareMetrics | null
    conversationId?: string | null
    modelId?: string | null
  }) => void
  onError: (message: string) => void
}

interface DeploymentCompareResult {
  success: boolean
  conversation_id: string
  response: string
  metrics?: CompareMetrics | null
  model_id?: string | null
}

export function serializeCompareMessages(
  messages: Array<{
    role: 'user' | 'assistant'
    content: string
    parts?: ChatContentBlock[]
    isStreaming?: boolean
  }>,
): SerializedMessage[] {
  return messages
    .filter((message) => !message.isStreaming)
    .map((message) => ({
      role: message.role,
      content: sanitizeChatContentForRequest(message.parts ?? message.content),
    }))
}

export async function streamAgentCompareTarget(
  target: CompareTargetConfig,
  messages: SerializedMessage[],
  options: {
    temperature?: number
    signal: AbortSignal
  },
  callbacks: AgentCompareCallbacks,
) {
  const response = await fetch(CHAT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: options.signal,
    body: JSON.stringify({
      messages,
      model: target.model,
      stream: true,
      temperature: options.temperature ?? 0.7,
    }),
  })

  if (!response.ok) {
    callbacks.onError(`Chat request failed (${response.status}): ${await response.text()}`)
    return
  }

  const reader = response.body?.getReader()
  if (!reader) {
    callbacks.onError('No response body')
    return
  }

  const decoder = new TextDecoder()
  let buffer = ''
  let currentEventType = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEventType = line.slice(7).trim()
        continue
      }

      if (!line.startsWith('data: ')) {
        continue
      }

      const data = line.slice(6)
      if (data === '[DONE]') {
        return
      }

      try {
        const parsed = JSON.parse(data) as Record<string, unknown>

        if (currentEventType) {
          handleAgentNamedEvent(currentEventType, parsed, callbacks)
          currentEventType = ''
          continue
        }

        const delta = (parsed.choices as Array<{ delta?: { content?: string } }> | undefined)?.[0]?.delta
        if (delta?.content) {
          callbacks.onToken(delta.content)
        }
      } catch {
        // Ignore malformed SSE payloads.
      }

      currentEventType = ''
    }
  }
}

function handleAgentNamedEvent(
  eventType: string,
  payload: Record<string, unknown>,
  callbacks: AgentCompareCallbacks,
) {
  switch (eventType) {
    case 'thinking':
      callbacks.onThinking((payload.content as string) ?? '')
      break
    case 'tool_start':
      callbacks.onToolStart(
        (payload.tool as string) ?? '',
        (payload.arguments as Record<string, unknown>) ?? {},
      )
      break
    case 'tool_end':
      callbacks.onToolEnd(
        (payload.tool as string) ?? '',
        Number(payload.duration_ms ?? 0),
      )
      break
    case 'reflection':
      callbacks.onReflection(
        Boolean(payload.is_ready),
        (payload.explanation as string) ?? '',
      )
      break
    case 'metrics':
      callbacks.onMetrics({
        confidence: (payload.confidence as string | null) ?? null,
        perplexity: asNumber(payload.perplexity),
        total_tokens: asNumber(payload.tokens),
        prompt_tokens: asNumber(
          (payload.usage as Record<string, unknown> | undefined)?.prompt_tokens,
        ),
        completion_tokens: asNumber(
          (payload.usage as Record<string, unknown> | undefined)?.completion_tokens,
        ),
      })
      break
    case 'confirmation_needed':
      callbacks.onError((payload.message as string) ?? 'Confirmation required')
      break
    case 'complete':
      callbacks.onComplete({
        metrics: (payload.metrics as CompareMetrics | undefined) ?? null,
        modelId: (payload.model_id as string | undefined) ?? null,
      })
      break
  }
}

export async function runDeploymentCompareTarget(
  target: CompareTargetConfig,
  userContent: string | ChatContentBlock[],
  conversationId: string | null,
  signal: AbortSignal,
  callbacks: DeploymentCompareCallbacks,
) {
  try {
    const requestContent = sanitizeChatContentForRequest(userContent)
    const isVisionLanguage =
      target.deploymentModality === 'vision-language' && Array.isArray(requestContent)

    if (isVisionLanguage) {
      const result = await mcpCall<DeploymentCompareResult>('host.chat_vlm', {
        deployment_id: target.deploymentId,
        messages: [{ role: 'user', content: requestContent }],
        ...(conversationId ? { conversation_id: conversationId } : {}),
        prefer_runtime_metrics: true,
      })
      callbacks.onComplete({
        response: result.response,
        metrics: result.metrics ?? null,
        conversationId: result.conversation_id,
        modelId: result.model_id ?? null,
      })
      return
    }

    if (Array.isArray(requestContent)) {
      callbacks.onError('Selected deployment does not support image attachments.')
      return
    }

    await streamDeploymentTextChat(
      {
        deployment_id: target.deploymentId,
        message: requestContent,
        conversation_id: conversationId,
        prefer_runtime_metrics: true,
        signal,
      },
      {
        onToken: callbacks.onToken,
        onComplete: (result) => {
          callbacks.onComplete({
            response: result.response,
            metrics: result.metrics ?? null,
            conversationId: result.conversation_id,
            modelId: result.model_id ?? null,
          })
        },
        onError: callbacks.onError,
      },
    )
  } catch (error) {
    callbacks.onError(error instanceof Error ? error.message : 'Deployment compare failed')
  }
}

function asNumber(value: unknown) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}
