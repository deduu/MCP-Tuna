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
        'cursor-pointer transition-colors hover:border-primary/50',
        isSelected && 'border-primary',
      )}
      onClick={onSelect}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="flex flex-col gap-1.5 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-muted-foreground">{shortId}</span>
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
            <p className="text-sm font-medium truncate">{modelName}</p>
            {deployment.name && (
              <p className="text-xs text-muted-foreground truncate">{deployment.model_path.split('/').pop() ?? deployment.model_path}</p>
            )}
            {lastUpdated && (
              <p className="text-[11px] text-muted-foreground">
                {deployment.status === 'stopped' ? 'Stopped' : 'Updated'} {lastUpdated}
              </p>
            )}
            <div className="flex items-center gap-1">
              <code className="text-xs text-muted-foreground truncate">{deployment.endpoint}</code>
              <button
                onClick={copyEndpoint}
                className="shrink-0 p-0.5 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              >
                <Clipboard className="h-3 w-3" />
              </button>
            </div>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            {deployment.status === 'stopped' && (
              <>
                <Button variant="outline" size="icon" className="h-7 w-7" onClick={(e) => handleRedeploy(e, 'mcp')} title="Redeploy as MCP">
                  <RotateCcw className="h-3 w-3" />
                </Button>
                <Button variant="ghost" size="icon" className="h-7 w-7" onClick={(e) => handleRedeploy(e, 'api')} title="Redeploy as API">
                  <span className="text-[10px] font-semibold">API</span>
                </Button>
              </>
            )}
            {deployment.status === 'running' && (
              <Button variant="outline" size="icon" className="h-7 w-7" onClick={handleStop}>
                <Square className="h-3 w-3" />
              </Button>
            )}
            <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive hover:text-destructive" onClick={handleUndeploy}>
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
