import { useEffect, useState } from 'react'
import { Rocket } from 'lucide-react'
import { useLocation, useNavigate } from 'react-router'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useDeployments } from '@/api/hooks/useDeployments'
import { DeploymentList } from './DeploymentList'
import { DeploymentDetail } from './DeploymentDetail'
import { DeployDialog, type DeployDialogInitialValues } from './DeployDialog'

type DeploymentsLocationState = {
  openDeployDialog?: boolean
  deployDialogType?: 'mcp' | 'api'
  deployInitialValues?: DeployDialogInitialValues | null
} | null

export function DeploymentsPage() {
  const [selectedDeploymentId, setSelectedDeploymentId] = useState<string | null>(null)
  const [deployDialogOpen, setDeployDialogOpen] = useState(false)
  const [deployDialogType, setDeployDialogType] = useState<'mcp' | 'api'>('mcp')
  const [deployInitialValues, setDeployInitialValues] = useState<DeployDialogInitialValues | null>(null)
  const location = useLocation()
  const navigate = useNavigate()

  const { data: deployments = [], isLoading } = useDeployments()

  const activeCount = deployments.filter((d) => d.status === 'running').length

  useEffect(() => {
    const state = location.state as DeploymentsLocationState
    if (!state?.openDeployDialog) return

    setDeployDialogType(state.deployDialogType ?? 'mcp')
    setDeployInitialValues(state.deployInitialValues ?? null)
    setDeployDialogOpen(true)
    navigate(location.pathname, { replace: true, state: null })
  }, [location.pathname, location.state, navigate])

  const openDeployDialog = (type: 'mcp' | 'api', initialValues: DeployDialogInitialValues | null = null) => {
    setDeployDialogType(type)
    setDeployInitialValues(initialValues)
    setDeployDialogOpen(true)
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Rocket className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-bold">Deployments</h1>
          {activeCount > 0 && (
            <Badge variant="success">{activeCount} active</Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button variant="default" onClick={() => openDeployDialog('mcp')}>
            Deploy as MCP
          </Button>
          <Button variant="outline" onClick={() => openDeployDialog('api')}>
            Deploy as API
          </Button>
        </div>
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <DeploymentList
            deployments={deployments}
            selectedId={selectedDeploymentId}
            onSelect={setSelectedDeploymentId}
            isLoading={isLoading}
          />
        </div>
        <div className="lg:col-span-2">
          {selectedDeploymentId ? (
            <DeploymentDetail deploymentId={selectedDeploymentId} />
          ) : (
            <div className="flex h-64 items-center justify-center rounded-xl border border-dashed text-muted-foreground">
              Select a deployment to view details
            </div>
          )}
        </div>
      </div>

      {/* Deploy dialog */}
      <DeployDialog
        open={deployDialogOpen}
        onClose={() => setDeployDialogOpen(false)}
        type={deployDialogType}
        initialValues={deployInitialValues}
      />
    </div>
  )
}
