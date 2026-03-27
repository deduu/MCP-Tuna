import { useQuery } from '@tanstack/react-query'
import { mcpCall } from '../client'
import type { DatasetInfo, TrainingJob } from '../types'
import { getDatasetLibraryRoots } from '@/lib/dataset-library-roots'

export interface EvalConfig {
  weights: Record<string, number>
  threshold: number
  language: string
  model?: string
  debug?: boolean
}

export interface DatasetListResult {
  datasets: DatasetInfo[]
  count: number
  scan_roots: string[]
  pruned_stale_records?: number
}

function resolveDatasetLibraryRoots(scanRoots?: string[]): string[] {
  return scanRoots && scanRoots.length > 0 ? scanRoots : getDatasetLibraryRoots()
}

export function useDatasetLibrary(scanRoots?: string[]) {
  const resolvedRoots = resolveDatasetLibraryRoots(scanRoots)

  return useQuery<DatasetListResult>({
    queryKey: ['datasets', 'library', ...resolvedRoots],
    queryFn: async () => {
      const result = await mcpCall<DatasetListResult>('dataset.list', { scan_roots: resolvedRoots })
      return {
        datasets: result.datasets ?? [],
        count: result.count ?? (result.datasets ?? []).length,
        scan_roots: result.scan_roots ?? resolvedRoots,
        pruned_stale_records:
          typeof result.pruned_stale_records === 'number' ? result.pruned_stale_records : 0,
      }
    },
    refetchInterval: 10_000,
    retry: 1,
  })
}

export function useDatasets(scanRoots?: string[]) {
  return useQuery<DatasetInfo[]>({
    queryKey: ['datasets', ...resolveDatasetLibraryRoots(scanRoots)],
    queryFn: async () => {
      const result = await mcpCall<DatasetListResult>('dataset.list', {
        scan_roots: resolveDatasetLibraryRoots(scanRoots),
      })
      return result.datasets ?? []
    },
    refetchInterval: 10_000,
    retry: 1,
  })
}

export function useDatasetStats(datasetId: string) {
  return useQuery({
    queryKey: ['datasets', 'stats', datasetId],
    queryFn: () => mcpCall<Record<string, unknown>>('dataset.info', { file_path: datasetId }),
    enabled: !!datasetId,
    staleTime: 30_000,
  })
}

export function useTechniques() {
  return useQuery<string[]>({
    queryKey: ['generate', 'techniques'],
    queryFn: async () => {
      const result = await mcpCall<{ techniques: string[] }>('generate.list_techniques')
      return result.techniques ?? []
    },
    staleTime: 5 * 60_000,
  })
}

export function useEvalMetrics() {
  return useQuery<string[]>({
    queryKey: ['evaluate', 'metrics'],
    queryFn: async () => {
      const result = await mcpCall<{ metrics: string[] }>('evaluate.list_metrics')
      return result.metrics ?? []
    },
    staleTime: 5 * 60_000,
    retry: 0,
  })
}

function normalizeDatasetJob(job: TrainingJob & Record<string, unknown>): TrainingJob {
  const rawStatus = typeof job.status === 'string' ? job.status : ''
  const status = rawStatus.startsWith('JobStatus.')
    ? rawStatus.slice('JobStatus.'.length).toLowerCase()
    : rawStatus

  return {
    ...job,
    status: status as TrainingJob['status'],
    technique: typeof job.technique === 'string' ? job.technique : job.trainer_type,
    trainer_type: typeof job.trainer_type === 'string' ? job.trainer_type : 'hf_blend',
  }
}

export function useDatasetBlendJobs(limit: number = 20) {
  return useQuery<TrainingJob[]>({
    queryKey: ['datasets', 'blend-jobs', limit],
    queryFn: async () => {
      const result = await mcpCall<{ jobs: TrainingJob[] }>('generate.list_hf_blend_jobs', { limit })
      return (result.jobs ?? [])
        .map((job) => normalizeDatasetJob(job as TrainingJob & Record<string, unknown>))
        .sort((a, b) => Date.parse(b.created_at ?? '') - Date.parse(a.created_at ?? ''))
    },
    refetchInterval: (query) => {
      const jobs = query.state.data ?? []
      return jobs.some((job) => job.status === 'running' || job.status === 'pending') ? 2_000 : 15_000
    },
    retry: 1,
  })
}

export function useEvalConfig() {
  return useQuery<EvalConfig>({
    queryKey: ['evaluate', 'config'],
    queryFn: async () => {
      const result = await mcpCall<{ config: EvalConfig }>('evaluate.get_config')
      return result.config
    },
    staleTime: 30_000,
    retry: 0,
  })
}
