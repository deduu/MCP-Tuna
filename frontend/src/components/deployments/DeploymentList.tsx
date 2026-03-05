import type { Deployment } from '@/api/types'
import { useStopDeployment, useUndeployment } from '@/api/hooks/useDeployments'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'sonner'
import { DeploymentCard } from './DeploymentCard'

interface DeploymentListProps {
  deployments: Deployment[]
  selectedId: string | null
  onSelect: (id: string) => void
  isLoading: boolean
}

export function DeploymentList({ deployments, selectedId, onSelect, isLoading }: DeploymentListProps) {
  const stopMutation = useStopDeployment()
  const undeployMutation = useUndeployment()

  const handleStop = (id: string) => {
    stopMutation.mutate(id, {
      onSuccess: () => toast.success('Deployment stopped'),
      onError: (err) => toast.error(`Failed to stop: ${err.message}`),
    })
  }

  const handleUndeploy = (id: string) => {
    undeployMutation.mutate(id, {
      onSuccess: () => toast.success('Deployment removed'),
      onError: (err) => toast.error(`Failed to undeploy: ${err.message}`),
    })
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-36 w-full rounded-xl" />
        ))}
      </div>
    )
  }

  if (deployments.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed p-8 text-center">
        <p className="text-sm font-medium text-muted-foreground">No deployments</p>
        <p className="text-xs text-muted-foreground">
          Deploy a model as an MCP server or API endpoint to get started.
        </p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {deployments.map((deployment) => (
        <DeploymentCard
          key={deployment.deployment_id}
          deployment={deployment}
          isSelected={selectedId === deployment.deployment_id}
          onSelect={() => onSelect(deployment.deployment_id)}
          onStop={() => handleStop(deployment.deployment_id)}
          onUndeploy={() => handleUndeploy(deployment.deployment_id)}
        />
      ))}
    </div>
  )
}
