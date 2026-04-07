const DEFAULT_GATEWAY_ORIGIN = 'http://127.0.0.1:8002'
const PROXIED_MCP_URL = '/mcp'
const PROXIED_HEALTH_URL = '/gateway-health'

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, '')
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)))
}

function preferProxy(): boolean {
  return typeof window !== 'undefined' && window.location.port === '5173'
}

function configuredGatewayOrigin(): string {
  const envOrigin = import.meta.env.VITE_MCP_HTTP_ORIGIN?.trim()
  return trimTrailingSlash(envOrigin || DEFAULT_GATEWAY_ORIGIN)
}

export function gatewayMcpUrls(): string[] {
  const directUrl = `${configuredGatewayOrigin()}/mcp`
  const preferred = preferProxy() ? [PROXIED_MCP_URL, directUrl] : [directUrl, PROXIED_MCP_URL]
  const explicitUrl = import.meta.env.VITE_MCP_URL?.trim()
  return unique(explicitUrl ? [explicitUrl, ...preferred] : preferred)
}

export function gatewayHealthUrls(): string[] {
  const directUrl = `${configuredGatewayOrigin()}/health`
  const preferred = preferProxy() ? [PROXIED_HEALTH_URL, directUrl] : [directUrl, PROXIED_HEALTH_URL]
  const explicitUrl = import.meta.env.VITE_GATEWAY_HEALTH_URL?.trim()
  return unique(explicitUrl ? [explicitUrl, ...preferred] : preferred)
}

export const FRONTEND_GATEWAY_START_COMMAND = 'uv run python scripts/run_gateway.py http --port 8002'
