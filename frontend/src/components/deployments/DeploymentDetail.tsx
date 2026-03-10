import { useDeployments, useDeploymentLogs } from '@/api/hooks/useDeployments'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { RefreshCw } from 'lucide-react'
import { LogViewer } from './LogViewer'
import { DeploymentChat } from './DeploymentChat'

interface DeploymentDetailProps {
  deploymentId: string
}

export function DeploymentDetail({ deploymentId }: DeploymentDetailProps) {
  const { data: deployments = [] } = useDeployments()
  const { data: logs = [], isLoading: logsLoading } = useDeploymentLogs(deploymentId, true)
  const statusMutation = useToolExecution()

  const deployment = deployments.find((d) => d.deployment_id === deploymentId)

  const refreshStatus = () => {
    statusMutation.mutate({ toolName: 'host.health', args: { deployment_id: deploymentId } })
  }

  if (!deployment) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-dashed text-muted-foreground">
        Deployment not found
      </div>
    )
  }

  const modelName = deployment.model_path.split('/').pop() ?? deployment.model_path

  return (
    <div className="flex flex-col gap-4">
      {/* Status header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Deployment Details</CardTitle>
            <Button variant="ghost" size="sm" onClick={refreshStatus} disabled={statusMutation.isPending}>
              <RefreshCw className={cn('h-4 w-4', statusMutation.isPending && 'animate-spin')} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
            <div>
              <dt className="text-muted-foreground">Deployment ID</dt>
              <dd className="font-mono">{deployment.deployment_id}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Status</dt>
              <dd>
                <Badge variant={deployment.status === 'running' ? 'success' : 'secondary'}>
                  {deployment.status}
                </Badge>
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Model</dt>
              <dd className="truncate">{modelName}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Type</dt>
              <dd>
                <Badge
                  className={cn(
                    deployment.type === 'mcp'
                      ? 'bg-[var(--color-ns-host)]/20 text-[var(--color-ns-host)] border-transparent'
                      : 'bg-primary/20 text-primary border-transparent',
                  )}
                >
                  {deployment.type === 'mcp' ? 'MCP Server' : 'API Endpoint'}
                </Badge>
              </dd>
            </div>
            <div className="col-span-2">
              <dt className="text-muted-foreground">Endpoint</dt>
              <dd className="font-mono text-xs break-all">{deployment.endpoint}</dd>
            </div>
            <div className="col-span-2">
              <dt className="text-muted-foreground">Model Path</dt>
              <dd className="font-mono text-xs break-all">{deployment.model_path}</dd>
            </div>
          </dl>

          {statusMutation.data && (
            <div className="mt-4 rounded-lg bg-secondary/50 p-3">
              <p className="text-xs font-medium text-muted-foreground mb-1">Live Status</p>
              <pre className="text-xs font-mono whitespace-pre-wrap">
                {JSON.stringify(statusMutation.data, null, 2)}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Logs */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Logs</CardTitle>
        </CardHeader>
        <CardContent>
          <LogViewer logs={logs} isLoading={logsLoading} />
        </CardContent>
      </Card>

      <DeploymentChat deployment={deployment} />
    </div>
  )
}
