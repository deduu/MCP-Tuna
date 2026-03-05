import { useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Radio, RefreshCw } from 'lucide-react'

export function GatewayConnection() {
  const { mutate, data, isPending, error } = useToolExecution()
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const hasRun = useRef(false)

  const runHealthCheck = () => {
    const start = performance.now()
    mutate(
      { toolName: 'system.health', args: {} },
      {
        onSuccess: () => setLatencyMs(Math.round(performance.now() - start)),
        onError: () => setLatencyMs(null),
      },
    )
  }

  useEffect(() => {
    if (!hasRun.current) {
      hasRun.current = true
      runHealthCheck()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const isConnected = data?.success === true && !error

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
          </div>
          <div className="flex items-center gap-3">
            {latencyMs !== null && isConnected && (
              <span className="text-xs text-muted-foreground font-mono">{latencyMs}ms</span>
            )}
            {isPending ? (
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
            Connection failed: {error.message}
          </p>
        )}

        <Button variant="outline" size="sm" onClick={runHealthCheck} disabled={isPending}>
          <RefreshCw className={`h-3.5 w-3.5 ${isPending ? 'animate-spin' : ''}`} />
          Test Connection
        </Button>
      </CardContent>
    </Card>
  )
}
