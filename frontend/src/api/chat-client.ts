import { useChatStore } from '@/stores/chat'
import type { TurnMetrics } from '@/stores/chat'

const CHAT_URL = '/v1/chat/completions'

interface ChatRequestOptions {
  model?: string
  temperature?: number
  selectedTools?: string[]
}

/**
 * Sends a chat request with SSE streaming and dispatches granular agent
 * events (thinking, tool_start, tool_end, reflection, phase, metrics)
 * to the chat store in real time.
 */
export async function sendChatMessage(
  userContent: string,
  options: ChatRequestOptions = {},
) {
  const store = useChatStore.getState()

  // Abort any in-flight stream
  store.abortController?.abort()

  const abortController = new AbortController()
  store.setStreaming(true, abortController)
  store.addUserMessage(userContent)
  const msgId = store.startAssistantMessage()

  // Build messages array from conversation history
  const messages = useChatStore
    .getState()
    .messages.filter((m) => !m.isStreaming)
    .map((m) => ({ role: m.role, content: m.content }))

  try {
    const response = await fetch(CHAT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: abortController.signal,
      body: JSON.stringify({
        messages,
        model: options.model ?? 'Auto',
        stream: true,
        temperature: options.temperature ?? 0.7,
        selected_tools: options.selectedTools,
      }),
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(`Chat request failed (${response.status}): ${text}`)
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let buffer = ''
    let currentEventType = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      // Keep the last potentially incomplete line
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        // Named SSE event type
        if (line.startsWith('event: ')) {
          currentEventType = line.slice(7).trim()
          continue
        }

        if (!line.startsWith('data: ')) continue
        const data = line.slice(6)

        if (data === '[DONE]') {
          store.finishAssistantMessage(msgId)
          store.setStreaming(false)
          return
        }

        try {
          const parsed = JSON.parse(data)

          // Custom agent events (named SSE events)
          if (currentEventType) {
            dispatchAgentEvent(msgId, currentEventType, parsed)
            currentEventType = ''
            continue
          }

          // Standard OpenAI chat chunk (token stream)
          const delta = parsed.choices?.[0]?.delta
          if (delta?.content) {
            store.appendToken(msgId, delta.content)
          }
        } catch {
          // Skip malformed lines
        }

        // Reset event type after processing data line
        currentEventType = ''
      }
    }

    // Stream ended without [DONE]
    store.finishAssistantMessage(msgId)
    store.setStreaming(false)
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      store.finishAssistantMessage(msgId)
      store.setStreaming(false)
      return
    }
    const message = err instanceof Error ? err.message : 'Unknown error'
    store.appendToken(msgId, `\n\n[Error: ${message}]`)
    store.finishAssistantMessage(msgId)
    store.setStreaming(false)
  }
}

function dispatchAgentEvent(
  msgId: string,
  eventType: string,
  data: Record<string, unknown>,
) {
  const store = useChatStore.getState()

  switch (eventType) {
    case 'thinking':
      store.addThinking(msgId, data.content as string)
      break

    case 'tool_start':
      store.addToolStart(
        msgId,
        data.tool as string,
        (data.arguments as Record<string, unknown>) ?? {},
      )
      break

    case 'tool_end':
      store.addToolEnd(
        msgId,
        data.tool as string,
        (data.duration_ms as number) ?? 0,
      )
      break

    case 'reflection':
      store.addReflection(
        msgId,
        data.is_ready as boolean,
        (data.explanation as string) ?? '',
      )
      break

    case 'phase':
      store.addPhase(
        msgId,
        data.phase as string,
        data.action as string,
      )
      break

    case 'metrics':
      store.addMetrics(msgId, {
        turn: data.turn as number,
        confidence: data.confidence as string | null,
        perplexity: data.perplexity as number | null,
        tokens: data.tokens as number | null,
        usage: data.usage as TurnMetrics['usage'],
      })
      break

    case 'complete':
      store.finishAssistantMessage(
        msgId,
        data.history as unknown[],
      )
      break
  }
}
