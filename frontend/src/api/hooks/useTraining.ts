import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mcpCall } from '../client'
import type {
  TrainingJob,
  HFSearchResult,
  RecommendResult,
  AutoPrescribeResult,
  LocalModelCandidate,
  DeploymentBrowseRoot,
  DeploymentBrowseResult,
} from '../types'

function normalizeTrainingJob(job: TrainingJob & Record<string, unknown>): TrainingJob {
  const trainerType = typeof job.trainer_type === 'string' ? job.trainer_type : undefined
  const technique = typeof job.technique === 'string'
    ? job.technique
    : trainerType

  return {
    ...job,
    trainer_type: trainerType,
    technique,
    dataset_path: typeof job.dataset_path === 'string' ? job.dataset_path : undefined,
  }
}

export function useTrainingJobs() {
  return useQuery<TrainingJob[]>({
    queryKey: ['training', 'jobs'],
    queryFn: async () => {
      const result = await mcpCall<{ jobs: TrainingJob[] } | TrainingJob>('finetune.get_status')
      if ('jobs' in result) return result.jobs.map((job) => normalizeTrainingJob(job as TrainingJob & Record<string, unknown>))
      if ('job_id' in result) return [normalizeTrainingJob(result as TrainingJob & Record<string, unknown>)]
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
    queryFn: async () => normalizeTrainingJob(
      await mcpCall<TrainingJob & Record<string, unknown>>('finetune.get_status', { job_id: jobId }),
    ),
    enabled: !!jobId,
    refetchInterval: 2_000,
  })
}

export function useAvailableModels() {
  return useQuery<string[]>({
    queryKey: ['training', 'models'],
    queryFn: async () => {
      const result = await mcpCall<{ models: Array<string | { id: string }> }>('validate.list_models')
      return (result.models ?? []).map((m) => (typeof m === 'string' ? m : m.id))
    },
    staleTime: 30_000,
  })
}

export function useLocalModelCandidates(query: string = '') {
  return useQuery<LocalModelCandidate[]>({
    queryKey: ['training', 'models', 'candidates', query],
    queryFn: async () => {
      const result = await mcpCall<{
        models: Array<string | { id: string; model_path?: string; usable_for?: string[] }>
      }>('validate.list_models', query.trim() ? { query } : {})

      return (result.models ?? []).map((item) => {
        if (typeof item === 'string') {
          return { id: item, model_path: item }
        }

        return {
          id: item.id,
          model_path: item.model_path ?? item.id,
          usable_for: item.usable_for,
        }
      })
    },
    staleTime: 30_000,
  })
}

export function useDeploymentBrowseRoots() {
  return useQuery<DeploymentBrowseRoot[]>({
    queryKey: ['file', 'deployment-browse', 'roots'],
    queryFn: async () => {
      const result = await mcpCall<{ roots: DeploymentBrowseRoot[] }>('file.list_deployment_roots')
      return result.roots ?? []
    },
    staleTime: 60_000,
  })
}

export function useDeploymentBrowseDir(rootId: string, path: string, enabled: boolean) {
  return useQuery<DeploymentBrowseResult>({
    queryKey: ['file', 'deployment-browse', rootId, path],
    queryFn: () => mcpCall<DeploymentBrowseResult>('file.browse_deployment_dir', { root_id: rootId, path }),
    enabled: enabled && !!rootId,
    staleTime: 10_000,
  })
}

type TrainParams = {
  technique: 'sft' | 'dpo' | 'grpo' | 'kto' | 'curriculum' | 'sequential'
  args: Record<string, unknown>
}

const TECHNIQUE_TOOLS: Record<string, string> = {
  sft: 'finetune.train_async',
  dpo: 'finetune.train_dpo_async',
  grpo: 'finetune.train_grpo_async',
  kto: 'finetune.train_kto_async',
  curriculum: 'finetune.train_curriculum_async',
  sequential: 'finetune.sequential_train_async',
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

export function useHFSearch(query: string, task: string = 'text-generation', enabled: boolean = false) {
  return useQuery<HFSearchResult>({
    queryKey: ['training', 'hf-search', query, task],
    queryFn: () => mcpCall<HFSearchResult>('validate.search_models', { query, task, limit: 20 }),
    enabled: enabled && query.length >= 2,
    staleTime: 60_000,
    retry: 1,
  })
}

export function useRecommendedModels(useCase: string = 'general') {
  return useQuery<RecommendResult>({
    queryKey: ['training', 'recommended', useCase],
    queryFn: () => mcpCall<RecommendResult>('validate.recommend_models', { use_case: useCase }),
    staleTime: 300_000,
  })
}

type AutoPrescribeParams = {
  dataset_path?: string
  dataset_row_count?: number
  dataset_avg_text_length?: number
  technique: string
  use_case?: string
}

export function useAutoSuggestModel() {
  return useMutation<AutoPrescribeResult, Error, AutoPrescribeParams>({
    mutationFn: (params) => mcpCall<AutoPrescribeResult>('system.auto_prescribe', params),
  })
}
