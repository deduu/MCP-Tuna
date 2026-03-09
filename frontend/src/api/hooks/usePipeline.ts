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
    queryFn: async () => [],
    staleTime: 30_000,
  })
}

export function usePipelineJobStatus(jobId: string) {
  return useQuery<PipelineJob>({
    queryKey: ['pipeline', 'job', jobId],
    queryFn: async () => ({
      job_id: jobId,
      status: 'completed',
    }),
    enabled: !!jobId,
    staleTime: 30_000,
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
    mutationFn: async (jobId: string) => ({
      success: false,
      job_id: jobId,
      message: 'Cancellation is not supported by current workflow tools.',
    }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline', 'jobs'] })
    },
  })
}
