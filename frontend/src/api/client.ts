import { gatewayHealthUrls, gatewayMcpUrls } from '@/lib/gateway'

const DEFAULT_TIMEOUT_MS = 15_000

interface MCPRequestOptions {
  timeoutMs?: number
  allowFallbackOnTimeout?: boolean
}

export interface GatewayHealthStatus {
  status: string
  sessions: number
  sse_connections: number
  tools: number
}

export class APIError extends Error {
  status?: number
  details?: unknown

  constructor(message: string, status?: number, details?: unknown) {
    super(message)
    this.name = 'APIError'
    this.status = status
    this.details = details
  }
}

function normalizeRequestError(error: unknown, timeoutMs: number, url: string) {
  if (error instanceof DOMException && error.name === 'AbortError') {
    return new APIError(`Gateway request timed out after ${Math.round(timeoutMs / 1000)}s`, undefined, { url })
  }
  if (error instanceof APIError) {
    return new APIError(error.message, error.status, { url, cause: error.details })
  }
  if (error instanceof Error) {
    return new APIError(error.message, undefined, { url })
  }
  return new APIError('Gateway request failed', undefined, { url })
}

async function fetchJsonWithFallback(
  urls: string[],
  options: RequestInit = {},
  requestOptions: MCPRequestOptions = {},
  failureLabel: string,
) {
  const timeoutMs = requestOptions.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const allowFallbackOnTimeout = requestOptions.allowFallbackOnTimeout ?? true
  const attempts: string[] = []

  for (const url of urls) {
    const controller = new AbortController()
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs)

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      })

      if (!response.ok) {
        throw new APIError(`Request failed: ${response.statusText}`, response.status)
      }

      return await response.json()
    } catch (error) {
      const timedOut = error instanceof DOMException && error.name === 'AbortError'
      const normalizedError = normalizeRequestError(error, timeoutMs, url)
      attempts.push(`${url} -> ${normalizedError.message}`)
      if (timedOut && !allowFallbackOnTimeout) {
        break
      }
    } finally {
      window.clearTimeout(timeoutId)
    }
  }

  throw new APIError(`${failureLabel}: ${attempts.join(' | ')}`)
}

async function mcpFetchJson(body: Record<string, unknown>, options: MCPRequestOptions = {}) {
  return fetchJsonWithFallback(
    gatewayMcpUrls(),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    options,
    'Could not reach the MCP gateway',
  )
}

async function fetchJson(urls: string[], options: RequestInit = {}, requestOptions: MCPRequestOptions = {}) {
  return fetchJsonWithFallback(urls, options, requestOptions, 'Could not reach the gateway')
}

export async function mcpCall<T = Record<string, unknown>>(
  toolName: string,
  args: Record<string, unknown> = {},
  options: MCPRequestOptions = {},
): Promise<T> {
  const json = await mcpFetchJson(
    {
      jsonrpc: '2.0',
      method: 'tools/call',
      params: { name: toolName, arguments: args },
      id: crypto.randomUUID(),
    },
    options,
  )

  if (json.error) {
    throw new APIError(json.error.message ?? 'MCP error', undefined, json.error)
  }

  const text = json.result?.content?.[0]?.text
  if (!text) {
    throw new APIError('Empty MCP response')
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    throw new APIError(text)
  }

  if (
    parsed &&
    typeof parsed === 'object' &&
    'success' in parsed &&
    (parsed as { success?: unknown }).success === false
  ) {
    const maybeError = (parsed as { error?: unknown }).error
    throw new APIError(
      typeof maybeError === 'string' && maybeError.trim()
        ? maybeError
        : 'Tool execution failed',
      undefined,
      parsed,
    )
  }

  return parsed as T
}

export async function mcpListTools(options: MCPRequestOptions = {}) {
  const json = await mcpFetchJson(
    {
      jsonrpc: '2.0',
      method: 'tools/list',
      id: crypto.randomUUID(),
    },
    options,
  )
  return json.result?.tools ?? []
}

export async function gatewayHealthCheck(options: MCPRequestOptions = {}): Promise<GatewayHealthStatus> {
  return fetchJson(gatewayHealthUrls(), { method: 'GET' }, options)
}
