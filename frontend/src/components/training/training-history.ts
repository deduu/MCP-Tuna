import type { TrainingJob } from '@/api/types'
import type { PipelineJob } from '@/api/hooks/usePipeline'
import type { BadgeProps } from '@/components/ui/badge'
import { formatDateTime, formatTimeAgo } from '@/lib/utils'

export type TrainingHistoryFilter = 'active' | 'finished' | 'all'
export type TrainingHistoryKind = 'training' | 'pipeline'

interface TrainingHistoryEntryBase {
  key: string
  kind: TrainingHistoryKind
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  title: string
  subtitle: string
  meta: string
  progress: number | null
  updatedLabel: string | null
  updatedExact: string | null
  searchText: string
  isActive: boolean
  sortTimestamp: number
}

export interface TrainingJobHistoryEntry extends TrainingHistoryEntryBase {
  kind: 'training'
  job: TrainingJob
}

export interface PipelineJobHistoryEntry extends TrainingHistoryEntryBase {
  kind: 'pipeline'
  job: PipelineJob
}

export type TrainingHistoryEntry = TrainingJobHistoryEntry | PipelineJobHistoryEntry

function getPathTail(value?: string | null, segments: number = 2) {
  if (!value) return 'Unknown'
  const normalized = value.replace(/\\/g, '/').split('/').filter(Boolean)
  if (normalized.length === 0) return value
  return normalized.slice(-segments).join('/')
}

function formatTechniqueLabel(value?: string | null) {
  if (!value) return 'Training'
  return value.replace(/_/g, ' ').toUpperCase()
}

function toTimestamp(...values: Array<string | undefined>) {
  for (const value of values) {
    const parsed = Date.parse(value ?? '')
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return 0
}

function isEntryActive(status: TrainingHistoryEntry['status']) {
  return status === 'running' || status === 'pending'
}

function buildTrainingEntry(job: TrainingJob): TrainingJobHistoryEntry {
  const updatedAt = job.completed_at ?? job.started_at ?? job.created_at
  const title = formatTechniqueLabel(job.technique ?? job.trainer_type)
  const subtitle = job.base_model
  const meta = job.dataset_path ? getPathTail(job.dataset_path) : getPathTail(job.output_dir)
  return {
    key: `training:${job.job_id}`,
    kind: 'training',
    status: job.status,
    title,
    subtitle,
    meta,
    progress: typeof job.progress?.percent_complete === 'number' ? job.progress.percent_complete : null,
    updatedLabel: formatTimeAgo(updatedAt),
    updatedExact: formatDateTime(updatedAt),
    searchText: [
      job.job_id,
      title,
      subtitle,
      meta,
      job.dataset_path,
      job.output_dir,
    ].filter(Boolean).join(' ').toLowerCase(),
    isActive: isEntryActive(job.status),
    sortTimestamp: toTimestamp(job.completed_at, job.started_at, job.created_at),
    job,
  }
}

function buildPipelineEntry(job: PipelineJob): PipelineJobHistoryEntry {
  const updatedAt = job.progress?.last_updated ?? job.completed_at ?? job.started_at ?? job.created_at
  const title = job.current_step ? `Pipeline: ${job.current_step}` : 'Pipeline run'
  const subtitle = job.steps?.length ? `${job.steps.length} configured steps` : 'Workflow training run'
  const meta = job.progress?.status_message
    ?? (job.steps?.slice(0, 2).join(' -> ') || 'Workflow job')

  return {
    key: `pipeline:${job.job_id}`,
    kind: 'pipeline',
    status: job.status,
    title,
    subtitle,
    meta,
    progress: typeof job.progress?.percent_complete === 'number' ? job.progress.percent_complete : null,
    updatedLabel: formatTimeAgo(updatedAt),
    updatedExact: formatDateTime(updatedAt),
    searchText: [
      job.job_id,
      title,
      subtitle,
      meta,
      job.current_step,
      ...(job.steps ?? []),
    ].filter(Boolean).join(' ').toLowerCase(),
    isActive: isEntryActive(job.status),
    sortTimestamp: toTimestamp(
      job.progress?.last_updated,
      job.completed_at,
      job.started_at,
      job.created_at,
    ),
    job,
  }
}

export function buildTrainingHistoryEntries(jobs: TrainingJob[], pipelineJobs: PipelineJob[]) {
  return [
    ...jobs.map(buildTrainingEntry),
    ...pipelineJobs.map(buildPipelineEntry),
  ].sort((a, b) => b.sortTimestamp - a.sortTimestamp)
}

export function filterTrainingHistoryEntries(
  entries: TrainingHistoryEntry[],
  filter: TrainingHistoryFilter,
  query: string,
) {
  const normalizedQuery = query.trim().toLowerCase()
  return entries.filter((entry) => {
    const matchesFilter =
      filter === 'all'
        ? true
        : filter === 'active'
          ? entry.isActive
          : !entry.isActive
    const matchesQuery = normalizedQuery
      ? entry.searchText.includes(normalizedQuery)
      : true

    return matchesFilter && matchesQuery
  })
}

export function paginateTrainingHistoryEntries(
  entries: TrainingHistoryEntry[],
  page: number,
  pageSize: number,
) {
  const totalItems = entries.length
  const totalPages = totalItems === 0 ? 1 : Math.ceil(totalItems / pageSize)
  const currentPage = Math.min(Math.max(page, 1), totalPages)
  const startIndex = (currentPage - 1) * pageSize
  const endIndex = Math.min(startIndex + pageSize, totalItems)

  return {
    page: currentPage,
    pageSize,
    totalItems,
    totalPages,
    startIndex,
    endIndex,
    items: entries.slice(startIndex, endIndex),
  }
}

export function getTrainingHistoryFilterCounts(entries: TrainingHistoryEntry[]) {
  return {
    active: entries.filter((entry) => entry.isActive).length,
    finished: entries.filter((entry) => !entry.isActive).length,
    all: entries.length,
  }
}

export function getHistoryStatusVariant(status: TrainingHistoryEntry['status']): BadgeProps['variant'] {
  switch (status) {
    case 'completed':
      return 'success'
    case 'running':
      return 'default'
    case 'pending':
      return 'secondary'
    case 'failed':
      return 'error'
    case 'cancelled':
      return 'outline'
    default:
      return 'secondary'
  }
}
