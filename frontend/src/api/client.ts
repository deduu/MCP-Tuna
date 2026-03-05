const GATEWAY_URL = '/mcp'

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

export async function mcpCall<T = Record<string, unknown>>(
  toolName: string,
  args: Record<string, unknown> = {},
): Promise<T> {
  const response = await fetch(GATEWAY_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: { name: toolName, arguments: args },
      id: crypto.randomUUID(),
    }),
  })

  if (!response.ok) {
    throw new APIError(`MCP call failed: ${response.statusText}`, response.status)
  }

  const json = await response.json()

  if (json.error) {
    throw new APIError(json.error.message ?? 'MCP error', undefined, json.error)
  }

  const text = json.result?.content?.[0]?.text
  if (!text) {
    throw new APIError('Empty MCP response')
  }

  return JSON.parse(text) as T
}

export async function mcpListTools() {
  const response = await fetch(GATEWAY_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0',
      method: 'tools/list',
      id: crypto.randomUUID(),
    }),
  })

  if (!response.ok) {
    throw new APIError(`Failed to list tools: ${response.statusText}`, response.status)
  }

  const json = await response.json()
  return json.result?.tools ?? []
}
