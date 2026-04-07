import { useEffect, useMemo, useState } from 'react'
import { Boxes, ImageIcon, Rocket, Server } from 'lucide-react'
import { useLocation, useNavigate } from 'react-router'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { getRedeployInitialValues, useDeployments } from '@/api/hooks/useDeployments'
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
  const apiCount = deployments.filter((d) => d.type === 'api').length
  const vlmCount = deployments.filter((d) => d.modality === 'vision-language').length
  const mcpCount = deployments.length - apiCount

  const summaryCards = useMemo(
    () => [
      {
        label: 'Running Now',
        value: `${activeCount}`,
        detail:
          activeCount > 0
            ? `${activeCount} deployment${activeCount === 1 ? '' : 's'} currently accepting requests.`
            : 'No active runtimes at the moment.',
        icon: Server,
      },
      {
        label: 'Endpoint Mix',
        value: `${mcpCount} MCP / ${apiCount} API`,
        detail:
          deployments.length > 0
            ? 'Split deployments between hosted MCP servers and direct API endpoints.'
            : 'Deploy as MCP or API once you are ready to expose a runtime.',
        icon: Boxes,
      },
      {
        label: 'Modalities',
        value: vlmCount > 0 ? `${vlmCount} VLM` : 'Text Only',
        detail:
          vlmCount > 0
            ? `${vlmCount} vision-language deployment${vlmCount === 1 ? '' : 's'} available for image-aware testing.`
            : 'Current deployments are configured for text-only inference.',
        icon: ImageIcon,
      },
    ],
    [activeCount, apiCount, deployments.length, mcpCount, vlmCount],
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

  return (
    <div className="relative flex h-[calc(100vh-3.5rem)] -m-6 flex-col gap-4 overflow-hidden px-6 py-4">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-44 bg-[radial-gradient(circle_at_top,rgba(59,130,246,0.16),transparent_62%)]" />

      <div className="relative rounded-2xl border border-border/90 bg-card p-5 shadow-[0_20px_54px_rgba(0,0,0,0.34)]">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-3">
              <Rocket className="h-6 w-6 text-primary" />
              <h1 className="text-2xl font-semibold tracking-tight">Deployments</h1>
              {activeCount > 0 && <Badge variant="success">{activeCount} active</Badge>}
            </div>
            <p className="max-w-3xl text-sm text-muted-foreground">
              Launch, inspect, and chat with local runtimes from one workspace. Use the left rail to
              track what is running and the detail pane to verify endpoints, logs, and live behavior.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button variant="default" onClick={() => openDeployDialog('mcp')} className="gap-2">
              Deploy as MCP
            </Button>
            <Button variant="outline" onClick={() => openDeployDialog('api')} className="gap-2">
              Deploy as API
            </Button>
          </div>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-3">
          {summaryCards.map((card) => (
            <div
              key={card.label}
              className="rounded-xl border border-border/90 bg-secondary p-4 shadow-sm shadow-black/20"
            >
              <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                <card.icon className="h-4 w-4 text-primary" />
                <span>{card.label}</span>
              </div>
              <p className="mt-3 text-2xl font-semibold tracking-tight">{card.value}</p>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{card.detail}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden">
        <div className="grid h-full min-h-0 grid-cols-1 gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
          <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-border/90 bg-card shadow-[0_20px_48px_rgba(0,0,0,0.28)]">
            <div className="border-b px-5 py-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="space-y-1">
                  <p className="text-sm font-medium">Deployment Inventory</p>
                  <p className="text-xs text-muted-foreground">
                    Select a runtime to review status, logs, and direct chat.
                  </p>
                </div>
                <Badge variant="outline">{deployments.length} total</Badge>
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
                  <p className="text-lg font-semibold">Select a deployment</p>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Pick a deployment from the left to inspect endpoint details, live logs, and the
                    embedded runtime chat surface.
                  </p>
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
      />
    </div>
  )
}
