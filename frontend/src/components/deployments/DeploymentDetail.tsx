import { useDeployments, useDeploymentLogs } from '@/api/hooks/useDeployments'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tab, TabList, TabPanel, Tabs } from '@/components/ui/tabs'
import { formatThinkingModeLabel } from '@/lib/thinking-mode'
import { cn, formatDateTime } from '@/lib/utils'
import { Info, Logs, MessageSquareText, RefreshCw, RotateCcw } from 'lucide-react'
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
                <CardDescription className="max-w-2xl break-all">
                  {deployment.endpoint}
                </CardDescription>
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
        <CardContent className="pt-5">
          <Tabs defaultValue="chat">
            <TabList className="mb-5 gap-6">
              <Tab value="chat" className="flex items-center gap-2 px-0">
                <MessageSquareText className="h-4 w-4" />
                Chat
              </Tab>
              <Tab value="details" className="flex items-center gap-2 px-0">
                <Info className="h-4 w-4" />
                Details
              </Tab>
              <Tab value="logs" className="flex items-center gap-2 px-0">
                <Logs className="h-4 w-4" />
                Logs
              </Tab>
            </TabList>

            <TabPanel value="chat">
              <DeploymentChat deployment={deployment} />
            </TabPanel>

            <TabPanel value="details">
              <div className="space-y-4">
                <dl className="grid gap-3 md:grid-cols-2">
                  <DetailField label="Name" value={modelName} />
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
                  <DetailField label="Endpoint" value={deployment.endpoint} mono muted />
                  <DetailField
                    label="Model"
                    value={deployment.model_path.split('/').pop() ?? deployment.model_path}
                  />
                  {deployment.modality === 'text' && (
                    <DetailField
                      label="Thinking Mode"
                      value={formatThinkingModeLabel(deployment.thinking_mode)}
                    />
                  )}
                  <DetailField label="Model Path" value={deployment.model_path} mono muted spanFull />
                  {deployment.adapter_path && (
                    <DetailField label="Adapter Path" value={deployment.adapter_path} mono muted spanFull />
                  )}
                  {deployment.created_at && (
                    <DetailField
                      label="Created"
                      value={formatDateTime(deployment.created_at) ?? deployment.created_at}
                    />
                  )}
                  {deployment.updated_at && (
                    <DetailField
                      label="Last Update"
                      value={formatDateTime(deployment.updated_at) ?? deployment.updated_at}
                    />
                  )}
                  {deployment.stopped_at && (
                    <DetailField
                      label="Stopped"
                      value={formatDateTime(deployment.stopped_at) ?? deployment.stopped_at}
                      spanFull
                    />
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
              </div>
            </TabPanel>

            <TabPanel value="logs">
              <LogViewer logs={logs} isLoading={logsLoading} />
            </TabPanel>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}

function DetailField({
  label,
  value,
  mono = false,
  muted = false,
  spanFull = false,
}: {
  label: string
  value: string
  mono?: boolean
  muted?: boolean
  spanFull?: boolean
}) {
  return (
    <div className={cn('rounded-xl border border-border/90 bg-secondary p-3', spanFull && 'md:col-span-2')}>
      <dt className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
        {label}
      </dt>
      <dd className={cn('mt-2 break-all text-sm', mono && 'font-mono text-xs', muted && 'text-muted-foreground')}>
        {value}
      </dd>
    </div>
  )
}
