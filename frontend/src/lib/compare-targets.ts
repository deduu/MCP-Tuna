import type { Deployment } from '@/api/types'
import type { CompareTargetConfig } from '@/stores/chatCompare'

export function shortDeploymentLabel(modelPath: string) {
  const normalized = modelPath.replace(/\\/g, '/')
  const parts = normalized.split('/')
  return parts[parts.length - 1] || modelPath
}

export function deploymentDisplayLabel(deployment: Deployment) {
  return deployment.name?.trim() || shortDeploymentLabel(deployment.model_path)
}

export function resolveCompareDeploymentTarget(
  target: CompareTargetConfig,
  deployments: Deployment[],
) {
  if (target.kind !== 'deployment') {
    return null
  }

  if (target.deploymentId) {
    const exact = deployments.find((deployment) => deployment.deployment_id === target.deploymentId)
    if (exact) {
      return exact
    }
  }

  const targetName = target.deploymentLabel ?? target.label
  if (!targetName) {
    return null
  }

  const matches = deployments.filter(
    (deployment) => deploymentDisplayLabel(deployment) === targetName,
  )
  return matches.length === 1 ? matches[0] : null
}

export function resolveCompareTarget(
  target: CompareTargetConfig,
  deployments: Deployment[],
): CompareTargetConfig {
  if (target.kind !== 'deployment') {
    return target
  }

  const deployment = resolveCompareDeploymentTarget(target, deployments)
  if (!deployment) {
    return target
  }

  const label = deploymentDisplayLabel(deployment)
  const shouldSyncLabel =
    !target.label ||
    target.label === target.deploymentLabel ||
    target.label === 'Deployment'

  return {
    ...target,
    deploymentId: deployment.deployment_id,
    deploymentLabel: label,
    deploymentModality: deployment.modality ?? 'text',
    label: shouldSyncLabel ? label : target.label,
  }
}
