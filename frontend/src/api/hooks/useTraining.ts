import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mcpCall } from '../client'
import type { TrainingJob } from '../types'

export function useTrainingJobs() {
  return useQuery<TrainingJob[]>({
    queryKey: ['training', 'jobs'],
    queryFn: async () => {
      const result = await mcpCall<{ jobs: TrainingJob[] } | TrainingJob>('finetune.get_status')
      if ('jobs' in result) return result.jobs
      if ('job_id' in result) return [result]
      return []
    },
    refetchInterval: (query) => {
      const jobs = query.state.data
      const hasRunning = jobs?.some((j) => j.status === 'running' || j.status === 'pending')
      return hasRunning ? 3_000 : 30_000
    },
    retry: 1,
  })
}

export function useTrainingJobStatus(jobId: string) {
  return useQuery<TrainingJob>({
    queryKey: ['training', 'job', jobId],
    queryFn: () => mcpCall<TrainingJob>('finetune.get_status', { job_id: jobId }),
    enabled: !!jobId,
    refetchInterval: 2_000,
  })
}

export function useAvailableModels() {
  return useQuery<string[]>({
    queryKey: ['training', 'models'],
    queryFn: async () => {
      const result = await mcpCall<{ models: string[] }>('finetune.list_models')
      return result.models ?? []
    },
    staleTime: 30_000,
  })
}

type TrainParams = {
  technique: 'sft' | 'dpo' | 'grpo' | 'kto' | 'sequential'
  args: Record<string, unknown>
}

const TECHNIQUE_TOOLS: Record<string, string> = {
  sft: 'finetune.train',
  dpo: 'finetune.train_dpo',
  grpo: 'finetune.train_grpo',
  kto: 'finetune.train_kto',
  sequential: 'finetune.sequential_train',
}

export function useStartTraining() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ technique, args }: TrainParams) =>
      mcpCall(TECHNIQUE_TOOLS[technique], args),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['training', 'jobs'] })
    },
  })
}

export function useCancelTraining() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => mcpCall('finetune.cancel', { job_id: jobId }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['training', 'jobs'] })
    },
  })
}
