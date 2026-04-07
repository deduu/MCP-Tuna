import { useDeployments, useDeploymentLogs } from '@/api/hooks/useDeployments'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn, formatDateTime } from '@/lib/utils'
import { RefreshCw, RotateCcw } from 'lucide-react'
import { LogViewer } from './LogViewer'
import { DeploymentChat } from './DeploymentChat'

interface DeploymentDetailProps {
  deploymentId: string
  onRedeploy?: (deploymentId: string, type: 'mcp' | 'api') => void
}

export function DeploymentDetail({ deploymentId, onRedeploy }: DeploymentDetailProps) {
  const { data: deployments = [] } = useDeployments()
  const { data: logs = [], isLoading: logsLoading } = useDeploymentLogs(deploymentId, true)
  const statusMutation = useToolExecution()

  const deployment = deployments.find((d) => d.deployment_id === deploymentId)

  const refreshStatus = () => {
    statusMutation.mutate({ toolName: 'host.health', args: { deployment_id: deploymentId } })
  }

  if (!deployment) {
    return (
      <div className="flex h-64 items-center justify-center rounded-2xl border border-border/90 bg-card text-muted-foreground shadow-[0_20px_48px_rgba(0,0,0,0.28)]">
        Deployment not found
      </div>
    )
  }

  const modelName = deployment.name?.trim() || deployment.model_path.split('/').pop() || deployment.model_path

  return (
    <div className="flex flex-col gap-4">
      <Card className="border-border/90 shadow-[0_20px_48px_rgba(0,0,0,0.28)]">
        <CardHeader className="gap-4 border-b pb-4">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge
                  className={cn(
                    deployment.type === 'mcp'
                      ? 'border-transparent bg-[var(--color-ns-host)]/20 text-[var(--color-ns-host)]'
                      : 'border-transparent bg-primary/20 text-primary',
                  )}
                >
                  {deployment.type === 'mcp' ? 'MCP Server' : 'API Endpoint'}
                </Badge>
                <Badge variant={deployment.status === 'running' ? 'success' : 'secondary'}>
                  {deployment.status}
                </Badge>
                <Badge variant="outline">
                  {deployment.modality === 'vision-language' ? 'Vision-Language' : 'Text'}
                </Badge>
              </div>

              <div className="space-y-1">
                <CardTitle className="text-xl">{modelName}</CardTitle>
                <p className="font-mono text-xs text-muted-foreground">{deployment.deployment_id}</p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={refreshStatus}
                disabled={statusMutation.isPending}
                className="gap-2"
              >
                <RefreshCw className={cn('h-4 w-4', statusMutation.isPending && 'animate-spin')} />
                Refresh
              </Button>
              {deployment.status === 'stopped' && onRedeploy && (
                <div className="flex flex-wrap items-center gap-2">
                  <Button variant="outline" size="sm" onClick={() => onRedeploy(deploymentId, 'mcp')} className="gap-2">
                    <RotateCcw className="h-4 w-4" />
                    Redeploy MCP
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => onRedeploy(deploymentId, 'api')}>
                    Redeploy API
                  </Button>
                </div>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 pt-4">
          <dl className="grid gap-3 md:grid-cols-2">
            <div className="rounded-xl border border-border/90 bg-secondary p-3">
              <dt className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                Name
              </dt>
              <dd className="mt-2 text-sm font-medium">{modelName}</dd>
            </div>

            <div className="rounded-xl border border-border/90 bg-secondary p-3">
              <dt className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                Status
              </dt>
              <dd className="mt-2">
                <Badge variant={deployment.status === 'running' ? 'success' : 'secondary'}>
                  {deployment.status}
                </Badge>
              </dd>
            </div>

            <div className="rounded-xl border border-border/90 bg-secondary p-3">
              <dt className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                Endpoint
              </dt>
              <dd className="mt-2 break-all font-mono text-xs text-muted-foreground">
                {deployment.endpoint}
              </dd>
            </div>

            <div className="rounded-xl border border-border/90 bg-secondary p-3">
              <dt className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                Model
              </dt>
              <dd className="mt-2 break-all text-sm">{deployment.model_path.split('/').pop() ?? deployment.model_path}</dd>
            </div>

            <div className="rounded-xl border border-border/90 bg-secondary p-3 md:col-span-2">
              <dt className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                Model Path
              </dt>
              <dd className="mt-2 break-all font-mono text-xs text-muted-foreground">
                {deployment.model_path}
              </dd>
            </div>

            {deployment.adapter_path && (
              <div className="rounded-xl border border-border/90 bg-secondary p-3 md:col-span-2">
                <dt className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                  Adapter Path
                </dt>
                <dd className="mt-2 break-all font-mono text-xs text-muted-foreground">
                  {deployment.adapter_path}
                </dd>
              </div>
            )}

            {deployment.created_at && (
              <div className="rounded-xl border border-border/90 bg-secondary p-3">
                <dt className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                  Created
                </dt>
                <dd className="mt-2 text-sm">{formatDateTime(deployment.created_at) ?? deployment.created_at}</dd>
              </div>
            )}

            {deployment.updated_at && (
              <div className="rounded-xl border border-border/90 bg-secondary p-3">
                <dt className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                  Last Update
                </dt>
                <dd className="mt-2 text-sm">{formatDateTime(deployment.updated_at) ?? deployment.updated_at}</dd>
              </div>
            )}

            {deployment.stopped_at && (
              <div className="rounded-xl border border-border/90 bg-secondary p-3 md:col-span-2">
                <dt className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                  Stopped
                </dt>
                <dd className="mt-2 text-sm">{formatDateTime(deployment.stopped_at) ?? deployment.stopped_at}</dd>
              </div>
            )}
          </dl>

          {statusMutation.data && (
            <div className="rounded-xl border border-border/90 bg-muted p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
              <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                Live Status
              </p>
              <pre className="mt-3 whitespace-pre-wrap break-all rounded-lg border border-border/80 bg-card p-3 text-xs font-mono">
                {JSON.stringify(statusMutation.data, null, 2)}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-border/90 shadow-[0_20px_48px_rgba(0,0,0,0.24)]">
        <CardHeader className="border-b pb-4">
          <CardTitle className="text-lg">Logs</CardTitle>
        </CardHeader>
        <CardContent className="pt-4">
          <LogViewer logs={logs} isLoading={logsLoading} />
        </CardContent>
      </Card>

      <DeploymentChat deployment={deployment} />
    </div>
  )
}
