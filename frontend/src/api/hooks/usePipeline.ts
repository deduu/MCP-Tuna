import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mcpCall } from '../client'

export interface PipelineJob {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  current_step?: string
  steps?: string[]
  progress?: number
  result?: Record<string, unknown>
  error?: string
  created_at?: string
}

export function usePipelineJobs() {
  return useQuery<PipelineJob[]>({
    queryKey: ['pipeline', 'jobs'],
    queryFn: async () => {
      const result = await mcpCall<{ jobs: PipelineJob[] }>('workflow.list_jobs')
      return result.jobs ?? []
    },
    refetchInterval: (query) => {
      const jobs = query.state.data
      const hasRunning = jobs?.some((j) => j.status === 'running' || j.status === 'pending')
      return hasRunning ? 5_000 : 30_000
    },
    retry: 1,
  })
}

export function usePipelineJobStatus(jobId: string) {
  return useQuery<PipelineJob>({
    queryKey: ['pipeline', 'job', jobId],
    queryFn: () => mcpCall<PipelineJob>('workflow.get_status', { job_id: jobId }),
    enabled: !!jobId,
    refetchInterval: 3_000,
  })
}

export function useRunPipeline() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (args: Record<string, unknown>) => mcpCall('workflow.run_pipeline', args),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline', 'jobs'] })
    },
  })
}

export function useRunFullPipeline() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (args: Record<string, unknown>) => mcpCall('workflow.full_pipeline', args),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline', 'jobs'] })
    },
  })
}

export function useCancelPipeline() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => mcpCall('workflow.cancel', { job_id: jobId }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline', 'jobs'] })
    },
  })
}
