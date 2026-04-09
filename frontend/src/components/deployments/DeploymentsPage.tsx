import { useEffect, useMemo, useState } from 'react'
import { Boxes, Eye, EyeOff, ImageIcon, Rocket, Server } from 'lucide-react'
import { useLocation, useNavigate } from 'react-router'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { DeployResult } from '@/api/hooks/useDeployments'
import { getRedeployInitialValues, useDeployments } from '@/api/hooks/useDeployments'
import { DeploymentList } from './DeploymentList'
import { DeploymentDetail } from './DeploymentDetail'
import {
  DeployDialog,
  type DeployDialogInitialValues,
  type DeployDialogSubmission,
} from './DeployDialog'
import { PendingDeploymentQueue } from './PendingDeploymentQueue'

type DeploymentsLocationState = {
  openDeployDialog?: boolean
  deployDialogType?: 'mcp' | 'api'
  deployInitialValues?: DeployDialogInitialValues | null
} | null

type PendingDeployment = DeployDialogSubmission & {
  deploymentId?: string
  phase: 'starting' | 'syncing'
}

export function DeploymentsPage() {
  const [selectedDeploymentId, setSelectedDeploymentId] = useState<string | null>(null)
  const [inventoryVisible, setInventoryVisible] = useState(true)
  const [deployDialogOpen, setDeployDialogOpen] = useState(false)
  const [deployDialogType, setDeployDialogType] = useState<'mcp' | 'api'>('mcp')
  const [deployInitialValues, setDeployInitialValues] = useState<DeployDialogInitialValues | null>(null)
  const [pendingDeployments, setPendingDeployments] = useState<PendingDeployment[]>([])
  const location = useLocation()
  const navigate = useNavigate()

  const { data: deployments = [], isLoading } = useDeployments()

  const activeCount = deployments.filter((d) => d.status === 'running').length
  const apiCount = deployments.filter((d) => d.type === 'api').length
  const vlmCount = deployments.filter((d) => d.modality === 'vision-language').length
  const mcpCount = deployments.length - apiCount
  const pendingCount = pendingDeployments.length

  const summaryPills = useMemo(
    () => [
      {
        label: activeCount > 0 ? `${activeCount} active` : pendingCount > 0 ? `${pendingCount} starting` : 'No active runtimes',
        icon: Server,
      },
      {
        label: `${mcpCount} MCP / ${apiCount} API`,
        icon: Boxes,
      },
      {
        label: vlmCount > 0 ? `${vlmCount} VLM` : 'Text only',
        icon: ImageIcon,
      },
    ],
    [activeCount, apiCount, mcpCount, pendingCount, vlmCount],
  )

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

  const handleDeployStart = (deployment: DeployDialogSubmission) => {
    setPendingDeployments((current) => [{ ...deployment, phase: 'starting' }, ...current])
  }

  const handleDeploySuccess = (requestId: string, result: DeployResult) => {
    setPendingDeployments((current) =>
      current.map((deployment) =>
        deployment.requestId === requestId
          ? {
              ...deployment,
              deploymentId: result.deployment_id,
              phase: 'syncing',
            }
          : deployment,
      ),
    )
  }

  const handleDeployError = (requestId: string) => {
    setPendingDeployments((current) =>
      current.filter((deployment) => deployment.requestId !== requestId),
    )
  }

  useEffect(() => {
    if (deployments.length === 0) {
      if (selectedDeploymentId !== null) {
        setSelectedDeploymentId(null)
      }
      return
    }

    const hasSelectedDeployment = deployments.some(
      (deployment) => deployment.deployment_id === selectedDeploymentId,
    )

    if (!hasSelectedDeployment) {
      setSelectedDeploymentId(deployments[0].deployment_id)
    }
  }, [deployments, selectedDeploymentId])

  useEffect(() => {
    if (pendingDeployments.length === 0 || deployments.length === 0) {
      return
    }

    const availableDeploymentIds = new Set(deployments.map((deployment) => deployment.deployment_id))
    const matchedPending = pendingDeployments.filter(
      (deployment) => deployment.deploymentId && availableDeploymentIds.has(deployment.deploymentId),
    )

    if (matchedPending.length === 0) {
      return
    }

    setPendingDeployments((current) =>
      current.filter(
        (deployment) => !deployment.deploymentId || !availableDeploymentIds.has(deployment.deploymentId),
      ),
    )

    if (!selectedDeploymentId && matchedPending[0].deploymentId) {
      setSelectedDeploymentId(matchedPending[0].deploymentId)
    }
  }, [deployments, pendingDeployments, selectedDeploymentId])

  return (
    <div className="relative flex h-[calc(100vh-3.5rem)] -m-6 flex-col gap-4 overflow-hidden px-6 py-4">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-44 bg-[radial-gradient(circle_at_top,rgba(59,130,246,0.16),transparent_62%)]" />

      <div className="relative rounded-2xl border border-border/90 bg-card px-5 py-4 shadow-[0_20px_54px_rgba(0,0,0,0.34)]">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-3">
              <Rocket className="h-6 w-6 text-primary" />
              <h1 className="text-2xl font-semibold tracking-tight">Deployments</h1>
              {activeCount > 0 && <Badge variant="success">{activeCount} active</Badge>}
              {pendingCount > 0 && <Badge variant="outline">{pendingCount} starting</Badge>}
            </div>
            <p className="max-w-3xl text-sm text-muted-foreground">
              Chat with deployed runtimes from one workspace. Open inventory only when you need to
              switch targets or inspect another runtime.
            </p>
            <div className="flex flex-wrap items-center gap-2 pt-1">
              {summaryPills.map((pill) => (
                <div
                  key={pill.label}
                  className="inline-flex items-center gap-2 rounded-full border border-border/80 bg-secondary/80 px-3 py-1.5 text-xs text-muted-foreground"
                >
                  <pill.icon className="h-3.5 w-3.5 text-primary" />
                  <span>{pill.label}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="ghost"
              onClick={() => setInventoryVisible((current) => !current)}
              className="gap-2"
            >
              {inventoryVisible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              {inventoryVisible ? 'Hide Inventory' : 'Show Inventory'}
            </Button>
            <Button variant="default" onClick={() => openDeployDialog('mcp')} className="gap-2">
              Deploy as MCP
            </Button>
            <Button variant="outline" onClick={() => openDeployDialog('api')} className="gap-2">
              Deploy as API
            </Button>
          </div>
        </div>
      </div>

      {pendingDeployments.length > 0 && <PendingDeploymentQueue items={pendingDeployments} />}

      <div className="relative min-h-0 flex-1 overflow-hidden">
        <div className={inventoryVisible ? 'grid h-full min-h-0 grid-cols-1 gap-4 lg:grid-cols-[320px_minmax(0,1fr)]' : 'h-full'}>
          {inventoryVisible && (
            <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-border/90 bg-card shadow-[0_20px_48px_rgba(0,0,0,0.28)]">
              <div className="border-b px-5 py-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Deployment Inventory</p>
                    <p className="text-xs text-muted-foreground">
                      Switch runtimes when you want a different chat target.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="outline">{deployments.length} total</Badge>
                    {pendingCount > 0 && <Badge variant="secondary">{pendingCount} starting</Badge>}
                  </div>
                </div>
              </div>

              <div className="min-h-0 overflow-y-auto p-4">
                <DeploymentList
                  deployments={deployments}
                  selectedId={selectedDeploymentId}
                  onSelect={setSelectedDeploymentId}
                  onRedeploy={(deploymentId, type) => {
                    const deployment = deployments.find((item) => item.deployment_id === deploymentId)
                    if (!deployment) return
                    openDeployDialog(type, getRedeployInitialValues(deployment))
                  }}
                  isLoading={isLoading}
                />
              </div>
            </section>
          )}

          <div className="min-h-0 overflow-y-auto pr-1">
            {selectedDeploymentId ? (
              <DeploymentDetail
                deploymentId={selectedDeploymentId}
                onRedeploy={(deploymentId, type) => {
                  const deployment = deployments.find((item) => item.deployment_id === deploymentId)
                  if (!deployment) return
                  openDeployDialog(type, getRedeployInitialValues(deployment))
                }}
              />
            ) : (
              <div className="flex h-full min-h-[320px] items-center justify-center rounded-2xl border border-border/90 bg-card p-8 text-center shadow-[0_20px_48px_rgba(0,0,0,0.28)]">
                <div className="max-w-md space-y-2">
                  <p className="text-lg font-semibold">
                    {pendingCount > 0 ? 'Deployment is starting' : 'Select a deployment'}
                  </p>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {pendingCount > 0
                      ? 'A new runtime is launching in the background. It will appear in the inventory automatically when it is ready, and you can keep using the rest of the workspace in the meantime.'
                      : 'Open inventory to choose a runtime, then the page stays focused on direct deployment chat.'}
                  </p>
                  {!inventoryVisible && deployments.length > 0 && (
                    <Button variant="outline" onClick={() => setInventoryVisible(true)} className="gap-2">
                      <Eye className="h-4 w-4" />
                      Show Inventory
                    </Button>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <DeployDialog
        open={deployDialogOpen}
        onClose={() => setDeployDialogOpen(false)}
        type={deployDialogType}
        initialValues={deployInitialValues}
        onDeployStart={handleDeployStart}
        onDeploySuccess={handleDeploySuccess}
        onDeployError={handleDeployError}
      />
    </div>
  )
}
