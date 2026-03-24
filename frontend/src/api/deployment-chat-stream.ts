export interface DeploymentStreamMetrics {
  prompt_tokens?: number | null
  completion_tokens?: number | null
  total_tokens?: number | null
  latency_ms?: number | null
  ttft_ms?: number | null
  output_tokens_per_second?: number | null
  estimated_cost_usd?: number | null
  confidence?: string | null
  perplexity?: number | null
}

export interface DeploymentStreamUsage {
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
}

export interface DeploymentStreamCompletePayload {
  success: boolean
  conversation_id: string
  deployment_id?: string | null
  response: string
  turns?: number
  metrics?: DeploymentStreamMetrics | null
  usage?: DeploymentStreamUsage | null
  model_id?: string | null
}

interface StreamDeploymentChatCallbacks {
  onToken: (token: string) => void
  onComplete: (payload: DeploymentStreamCompletePayload) => void
  onError: (message: string) => void
}

interface StreamDeploymentChatRequest {
  message: string
  deployment_id?: string | null
  endpoint?: string | null
  model_path?: string | null
  adapter_path?: string | null
  conversation_id?: string | null
  max_new_tokens?: number
  temperature?: number
  top_p?: number
  top_k?: number
  system_prompt?: string | null
  prefer_runtime_metrics?: boolean
  signal?: AbortSignal
}

const STREAM_URL = '/mcp/chat/stream'

export async function streamDeploymentTextChat(
  payload: StreamDeploymentChatRequest,
  callbacks: StreamDeploymentChatCallbacks,
) {
  const response = await fetch(STREAM_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: payload.signal,
  })

  if (!response.ok) {
    callbacks.onError(`Deployment chat failed (${response.status}): ${await response.text()}`)
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
    if (done) {
      return
    }

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

      try {
        const payloadData = JSON.parse(line.slice(6)) as Record<string, unknown>
        if (currentEventType === 'token') {
          const token = payloadData.content
          if (typeof token === 'string' && token) {
            callbacks.onToken(token)
          }
        } else if (currentEventType === 'complete') {
          callbacks.onComplete(payloadData as unknown as DeploymentStreamCompletePayload)
          return
        } else if (currentEventType === 'error') {
          callbacks.onError(
            typeof payloadData.error === 'string' ? payloadData.error : 'Deployment chat failed',
          )
          return
        }
      } catch {
        // Ignore malformed SSE payloads.
      } finally {
        currentEventType = ''
      }
    }
  }
}
