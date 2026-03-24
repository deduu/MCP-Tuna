const GATEWAY_URL = '/mcp'
const GATEWAY_HEALTH_URL = '/gateway-health'
const DEFAULT_TIMEOUT_MS = 15_000

interface MCPRequestOptions {
  timeoutMs?: number
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

async function mcpFetchJson(body: Record<string, unknown>, options: MCPRequestOptions = {}) {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(GATEWAY_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })

    if (!response.ok) {
      throw new APIError(`MCP call failed: ${response.statusText}`, response.status)
    }

    return await response.json()
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new APIError(`Gateway request timed out after ${Math.round(timeoutMs / 1000)}s`)
    }
    throw error
  } finally {
    window.clearTimeout(timeoutId)
  }
}

async function fetchJson(url: string, options: RequestInit = {}, requestOptions: MCPRequestOptions = {}) {
  const timeoutMs = requestOptions.timeoutMs ?? DEFAULT_TIMEOUT_MS
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
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new APIError(`Gateway request timed out after ${Math.round(timeoutMs / 1000)}s`)
    }
    throw error
  } finally {
    window.clearTimeout(timeoutId)
  }
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
  return fetchJson(GATEWAY_HEALTH_URL, { method: 'GET' }, options)
}
