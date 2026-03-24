import type { DatasetInfo, Deployment, TrainingJob } from '@/api/types'
import type { PipelineJob } from '@/api/hooks/usePipeline'
import type { BadgeProps } from '@/components/ui/badge'

export function getStatusBadgeVariant(
  status?: string,
): BadgeProps['variant'] {
  switch (status) {
    case 'completed':
    case 'running':
    case 'green':
      return 'success'
    case 'pending':
    case 'yellow':
      return 'warning'
    case 'failed':
    case 'cancelled':
    case 'red':
      return 'error'
    default:
      return 'secondary'
  }
}

export function formatTechniqueLabel(value?: string | null) {
  if (!value) return 'Training'
  return value.replace(/_/g, ' ').toUpperCase()
}

export function formatJobTitle(job: TrainingJob) {
  return formatTechniqueLabel(job.technique ?? job.trainer_type)
}

export function formatPipelineTitle(job: PipelineJob) {
  return job.current_step
    ? `Pipeline: ${job.current_step}`
    : 'Pipeline run'
}

export function formatDeploymentTitle(deployment: Deployment) {
  return deployment.name
    ?? getPathTail(deployment.adapter_path ?? deployment.model_path)
}

export function formatDatasetTitle(dataset: DatasetInfo) {
  return dataset.dataset_id || getPathTail(dataset.file_path)
}

export function getPathTail(value?: string | null, segments: number = 2) {
  if (!value) return 'Unknown'
  const normalized = value.replace(/\\/g, '/').split('/').filter(Boolean)
  if (normalized.length === 0) return value
  return normalized.slice(-segments).join('/')
}
