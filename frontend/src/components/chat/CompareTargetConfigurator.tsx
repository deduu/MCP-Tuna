import { Trash2 } from 'lucide-react'
import type { Deployment } from '@/api/types'
import { AVAILABLE_CHAT_MODELS } from './chat-model-options'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { CompareSession, CompareTargetConfig } from '@/stores/chatCompare'

interface CompareTargetConfiguratorProps {
  session: CompareSession
  index: number
  baselineTargetId: string | null
  runningDeployments: Deployment[]
  onSetBaseline: (targetId: string | null) => void
  onUpdate: (targetId: string, patch: Partial<CompareTargetConfig>) => void
  onRemove: (targetId: string) => void
}

export function CompareTargetConfigurator({
  session,
  index,
  baselineTargetId,
  runningDeployments,
  onSetBaseline,
  onUpdate,
  onRemove,
}: CompareTargetConfiguratorProps) {
  const selectedDeployment = runningDeployments.find(
    (deployment) => deployment.deployment_id === session.target.deploymentId,
  )

  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="text-sm">Target {index + 1}</CardTitle>
            <p className="mt-1 text-xs text-muted-foreground">
              {session.target.kind === 'agent'
                ? 'Managed provider model with MCP tool access.'
                : 'Running deployment compared through hosted or local runtime chat.'}
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={() => onRemove(session.target.id)}>
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-3 md:grid-cols-[120px_minmax(0,1fr)]">
          <label className="text-xs font-medium text-muted-foreground">Type</label>
          <select
            value={session.target.kind}
            onChange={(event) => {
              const kind = event.target.value as CompareTargetConfig['kind']
              if (kind === 'agent') {
                const firstModel = AVAILABLE_CHAT_MODELS[0]
                onUpdate(session.target.id, {
                  kind,
                  model: firstModel.id,
                  label: firstModel.label,
                  deploymentId: null,
                  deploymentLabel: null,
                  deploymentModality: 'text',
                })
                return
              }

              const firstDeployment = runningDeployments[0]
              onUpdate(session.target.id, {
                kind,
                model: undefined,
                label: firstDeployment ? shortDeploymentLabel(firstDeployment.model_path) : 'Deployment',
                deploymentId: firstDeployment?.deployment_id ?? null,
                deploymentLabel: firstDeployment ? shortDeploymentLabel(firstDeployment.model_path) : null,
                deploymentModality: firstDeployment?.modality ?? 'text',
              })
            }}
            className="h-9 rounded-md border border-input bg-background px-3 text-xs text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <option value="agent">Agent</option>
            <option value="deployment">Deployment</option>
          </select>
        </div>

        <div className="grid gap-3 md:grid-cols-[120px_minmax(0,1fr)]">
          <label className="text-xs font-medium text-muted-foreground">Target</label>
          {session.target.kind === 'agent' ? (
            <select
              value={session.target.model}
              onChange={(event) => {
                const selected = AVAILABLE_CHAT_MODELS.find((model) => model.id === event.target.value)
                onUpdate(session.target.id, {
                  model: event.target.value,
                  label: selected?.label ?? event.target.value,
                })
              }}
              className="h-9 rounded-md border border-input bg-background px-3 text-xs text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {AVAILABLE_CHAT_MODELS.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          ) : (
            <select
              value={session.target.deploymentId ?? ''}
              onChange={(event) => {
                const deployment = runningDeployments.find(
                  (candidate) => candidate.deployment_id === event.target.value,
                )
                onUpdate(session.target.id, {
                  deploymentId: event.target.value || null,
                  label: deployment ? shortDeploymentLabel(deployment.model_path) : session.target.label,
                  deploymentLabel: deployment ? shortDeploymentLabel(deployment.model_path) : null,
                  deploymentModality: deployment?.modality ?? 'text',
                })
              }}
              disabled={runningDeployments.length === 0}
              className="h-9 rounded-md border border-input bg-background px-3 text-xs text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            >
              {runningDeployments.length === 0 ? (
                <option value="">No running deployments</option>
              ) : (
                runningDeployments.map((deployment) => (
                  <option key={deployment.deployment_id} value={deployment.deployment_id}>
                    {shortDeploymentLabel(deployment.model_path)}
                    {deployment.modality === 'vision-language' ? ' (VLM)' : ''}
                  </option>
                ))
              )}
            </select>
          )}
        </div>

        <div className="grid gap-3 md:grid-cols-[120px_minmax(0,1fr)]">
          <label className="text-xs font-medium text-muted-foreground">Label</label>
          <input
            value={session.target.label}
            onChange={(event) => onUpdate(session.target.id, { label: event.target.value })}
            className="h-9 rounded-md border border-input bg-background px-3 text-xs text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
        </div>

        <div className="flex items-center justify-between rounded-md border border-border/70 bg-secondary/20 px-3 py-2">
          <div className="text-xs text-muted-foreground">
            {session.target.kind === 'agent'
              ? 'Baseline deltas compare MCP agent behavior and tool usage.'
              : selectedDeployment
                ? `Deployment compare uses ${selectedDeployment.type === 'api' ? 'the hosted runtime' : 'the live local runtime'} with direct metrics collection.`
                : 'Pick a running deployment before sending compare prompts.'}
          </div>
          <Button
            variant={baselineTargetId === session.target.id ? 'default' : 'outline'}
            size="sm"
            onClick={() => onSetBaseline(session.target.id)}
          >
            Baseline
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function shortDeploymentLabel(modelPath: string) {
  const normalized = modelPath.replace(/\\/g, '/')
  const parts = normalized.split('/')
  return parts[parts.length - 1] || modelPath
}
