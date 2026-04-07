import type { Deployment } from '@/api/types'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn, formatTimeAgo } from '@/lib/utils'
import { RotateCcw, Square, Trash2, Clipboard } from 'lucide-react'
import { toast } from 'sonner'

interface DeploymentCardProps {
  deployment: Deployment
  isSelected: boolean
  onSelect: () => void
  onRedeploy: (type: 'mcp' | 'api') => void
  onStop: () => void
  onUndeploy: () => void
}

export function DeploymentCard({ deployment, isSelected, onSelect, onRedeploy, onStop, onUndeploy }: DeploymentCardProps) {
  const modelName = deployment.name?.trim() || deployment.model_path.split('/').pop() || deployment.model_path
  const shortId = deployment.deployment_id.slice(0, 8)
  const lastUpdated = formatTimeAgo(deployment.updated_at ?? deployment.created_at)
  const baseModelName = deployment.model_path.split('/').pop() ?? deployment.model_path

  const copyEndpoint = (e: React.MouseEvent) => {
    e.stopPropagation()
    navigator.clipboard.writeText(deployment.endpoint)
    toast.success('Endpoint copied to clipboard')
  }

  const handleStop = (e: React.MouseEvent) => {
    e.stopPropagation()
    onStop()
  }

  const handleUndeploy = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (window.confirm('Are you sure you want to undeploy this model? This action cannot be undone.')) {
      onUndeploy()
    }
  }

  const handleRedeploy = (e: React.MouseEvent, type: 'mcp' | 'api') => {
    e.stopPropagation()
    onRedeploy(type)
  }

  return (
    <Card
      className={cn(
        'cursor-pointer border-border/90 bg-card shadow-[0_12px_28px_rgba(0,0,0,0.18)] transition-all hover:border-primary/45 hover:bg-accent/30 hover:shadow-[0_16px_34px_rgba(0,0,0,0.24)]',
        isSelected && 'border-primary/50 bg-primary/10 shadow-[0_18px_38px_rgba(59,130,246,0.12)]',
      )}
      onClick={onSelect}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full border border-border/80 bg-muted px-2 py-0.5 text-[10px] font-mono text-muted-foreground">
                {shortId}
              </span>
              <Badge
                className={cn(
                  deployment.type === 'mcp'
                    ? 'bg-[var(--color-ns-host)]/20 text-[var(--color-ns-host)] border-transparent'
                    : 'bg-primary/20 text-primary border-transparent',
                )}
              >
                {deployment.type === 'mcp' ? 'MCP' : 'API'}
              </Badge>
              <Badge variant={deployment.status === 'running' ? 'success' : 'secondary'}>
                {deployment.status}
              </Badge>
              <Badge variant="outline">
                {deployment.modality === 'vision-language' ? 'VLM' : 'Text'}
              </Badge>
            </div>

            <div className="space-y-1">
              <p className="truncate text-sm font-medium">{modelName}</p>
              {deployment.name && (
                <p className="truncate text-xs text-muted-foreground">{baseModelName}</p>
              )}
            </div>

            {lastUpdated && (
              <p className="text-[11px] text-muted-foreground">
                {deployment.status === 'stopped' ? 'Stopped' : 'Updated'} {lastUpdated}
              </p>
            )}

            <div className="rounded-lg border border-border/80 bg-muted px-3 py-2">
              <p className="text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
                Endpoint
              </p>
              <div className="mt-1 flex items-center gap-2">
                <code className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
                  {deployment.endpoint}
                </code>
                <button
                  onClick={copyEndpoint}
                  className="shrink-0 rounded p-1 text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
                  title="Copy endpoint"
                >
                  <Clipboard className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {deployment.status === 'stopped' && (
              <>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8 border-border/90 bg-secondary"
                  onClick={(e) => handleRedeploy(e, 'mcp')}
                  title="Redeploy as MCP"
                >
                  <RotateCcw className="h-3 w-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-muted-foreground hover:bg-muted hover:text-foreground"
                  onClick={(e) => handleRedeploy(e, 'api')}
                  title="Redeploy as API"
                >
                  <span className="text-[10px] font-semibold">API</span>
                </Button>
              </>
            )}
            {deployment.status === 'running' && (
              <Button
                variant="outline"
                size="icon"
                className="h-8 w-8 border-border/90 bg-secondary"
                onClick={handleStop}
                title="Stop deployment"
              >
                <Square className="h-3 w-3" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-destructive hover:bg-destructive/10 hover:text-destructive"
              onClick={handleUndeploy}
              title="Undeploy"
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
