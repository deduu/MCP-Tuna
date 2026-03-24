import { useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { gatewayHealthCheck } from '@/api/client'
import { Radio, RefreshCw } from 'lucide-react'

const CONNECTION_TIMEOUT_MS = 5_000
type ConnectionStatus = 'checking' | 'connected' | 'disconnected'

export function GatewayConnection() {
  const [status, setStatus] = useState<ConnectionStatus>('checking')
  const [toolCount, setToolCount] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const hasRun = useRef(false)
  const activeRequest = useRef(0)

  const runHealthCheck = async () => {
    const requestId = activeRequest.current + 1
    activeRequest.current = requestId
    setStatus('checking')
    setError(null)
    const start = performance.now()

    try {
      const health = await gatewayHealthCheck({ timeoutMs: CONNECTION_TIMEOUT_MS })
      if (activeRequest.current !== requestId) return

      setLatencyMs(Math.round(performance.now() - start))
      setToolCount(health.tools)
      setStatus(health.status === 'healthy' ? 'connected' : 'disconnected')
      if (health.status !== 'healthy') {
        setError(`Gateway returned status: ${health.status}`)
      }
    } catch (err) {
      if (activeRequest.current !== requestId) return

      setLatencyMs(null)
      setToolCount(null)
      setStatus('disconnected')
      setError(err instanceof Error ? err.message : 'Connection failed')
    }
  }

  useEffect(() => {
    if (!hasRun.current) {
      hasRun.current = true
      runHealthCheck()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const isConnected = status === 'connected'
  const isChecking = status === 'checking'

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Radio className="h-4 w-4" />
          Gateway Connection
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Gateway URL</p>
            <p className="font-mono text-sm">/mcp</p>
            {typeof toolCount === 'number' && isConnected && (
              <p className="text-xs text-muted-foreground">{toolCount} tools discovered</p>
            )}
          </div>
          <div className="flex items-center gap-3">
            {latencyMs !== null && isConnected && (
              <span className="text-xs text-muted-foreground font-mono">{latencyMs}ms</span>
            )}
            {isChecking ? (
              <Badge variant="secondary">Checking...</Badge>
            ) : isConnected ? (
              <Badge variant="success">Connected</Badge>
            ) : (
              <Badge variant="error">Disconnected</Badge>
            )}
          </div>
        </div>

        {error && (
          <p className="text-sm text-destructive">
            Connection failed: {error}
          </p>
        )}

        <Button variant="outline" size="sm" onClick={() => void runHealthCheck()} disabled={isChecking}>
          <RefreshCw className={`h-3.5 w-3.5 ${isChecking ? 'animate-spin' : ''}`} />
          Test Connection
        </Button>
      </CardContent>
    </Card>
  )
}
