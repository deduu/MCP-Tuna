import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { mcpCall } from '../client'

interface WorkflowJobPayload {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at?: string
  started_at?: string
  completed_at?: string
  error?: string
  result?: Record<string, unknown>
  progress?: {
    current_step?: number
    max_steps?: number
    current_epoch?: number
    max_epochs?: number
    loss?: number
    learning_rate?: number
    eval_loss?: number
    grad_norm?: number
    eta_seconds?: number
    percent_complete?: number
    gpu_memory_used_gb?: number
    gpu_memory_total_gb?: number
    current_stage?: string
    last_updated?: string
    status_message?: string
    stage_current?: number
    stage_total?: number
    stage_unit?: string
    log_history?: Array<{
      loss?: number
      learning_rate?: number
      epoch?: number
      step?: number
    }>
  }
  steps?: string[]
  config_summary?: {
    steps?: string[]
  }
}

interface WorkflowJobListResponse {
  success: boolean
  jobs?: WorkflowJobPayload[]
}

export interface PipelineJob {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  current_step?: string
  steps?: string[]
  progress?: WorkflowJobPayload['progress']
  result?: Record<string, unknown>
  error?: string
  created_at?: string
  started_at?: string
  completed_at?: string
}

function mapWorkflowJob(job: WorkflowJobPayload): PipelineJob {
  return {
    job_id: job.job_id,
    status: job.status,
    current_step: job.progress?.current_stage,
    steps: Array.isArray(job.steps)
      ? job.steps
      : Array.isArray(job.config_summary?.steps)
        ? job.config_summary?.steps
        : [],
    progress: job.progress,
    result: job.result,
    error: job.error,
    created_at: job.created_at,
    started_at: job.started_at,
    completed_at: job.completed_at,
  }
}

async function fetchPipelineJobStatus(jobId: string) {
  const response = await mcpCall<WorkflowJobPayload>('workflow.job_status', { job_id: jobId })
  return mapWorkflowJob(response)
}

function isActivePipelineJobStatus(status: PipelineJob['status']) {
  return status === 'running' || status === 'pending'
}

async function waitForPipelineJobToStop(jobId: string, timeoutMs: number = 30_000) {
  const startedAt = Date.now()

  while (Date.now() - startedAt < timeoutMs) {
    const job = await fetchPipelineJobStatus(jobId)
    if (!isActivePipelineJobStatus(job.status)) {
      return job
    }

    await new Promise((resolve) => window.setTimeout(resolve, 1_000))
  }

  throw new Error('Workflow job is still stopping. Try again in a few seconds.')
}

export function usePipelineJobs(limit: number = 50) {
  return useQuery<PipelineJob[]>({
    queryKey: ['pipeline', 'jobs', limit],
    queryFn: async () => {
      const response = await mcpCall<WorkflowJobListResponse>('workflow.list_jobs', { limit })
      return (response.jobs ?? []).map(mapWorkflowJob)
    },
    staleTime: 1_000,
    refetchInterval: (query) => {
      const jobs = query.state.data ?? []
      return jobs.some((job) => job.status === 'running' || job.status === 'pending') ? 1_000 : 10_000
    },
    refetchIntervalInBackground: true,
  })
}

export function usePipelineJobStatus(jobId: string) {
  return useQuery<PipelineJob>({
    queryKey: ['pipeline', 'job', jobId],
    queryFn: async () => fetchPipelineJobStatus(jobId),
    enabled: !!jobId,
    staleTime: 1_000,
    refetchInterval: (query) => {
      const job = query.state.data
      return job && (job.status === 'running' || job.status === 'pending') ? 1_000 : false
    },
    refetchIntervalInBackground: true,
  })
}

export function useRunPipeline() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (args: Record<string, unknown>) => mcpCall('workflow.run_pipeline_async', args),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline', 'jobs'] })
    },
  })
}

export function useRunFullPipeline() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (args: Record<string, unknown>) => mcpCall('workflow.full_pipeline_async', args),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline', 'jobs'] })
    },
  })
}

export function useCancelPipeline() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (jobId: string) => mcpCall('workflow.cancel_job', { job_id: jobId }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline', 'jobs'] })
    },
  })
}

export function useRemovePipelineJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ jobId, status }: { jobId: string; status: PipelineJob['status'] }) => {
      if (isActivePipelineJobStatus(status)) {
        const cancelResult = await mcpCall<{ success?: boolean; error?: string }>('workflow.cancel_job', {
          job_id: jobId,
        })
        if (cancelResult.success === false) {
          throw new Error(cancelResult.error ?? 'Cancel failed')
        }

        await waitForPipelineJobToStop(jobId)
      }

      const deleteResult = await mcpCall<{ success?: boolean; error?: string }>('workflow.delete_job', {
        job_id: jobId,
      })
      if (deleteResult.success === false) {
        throw new Error(deleteResult.error ?? 'Delete failed')
      }

      return deleteResult
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline', 'jobs'] })
    },
  })
}
