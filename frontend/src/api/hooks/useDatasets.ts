import { useQuery } from '@tanstack/react-query'
import { mcpCall } from '../client'
import type { DatasetInfo } from '../types'

export interface EvalConfig {
  weights: Record<string, number>
  threshold: number
  language: string
  model?: string
  debug?: boolean
}

export function useDatasets() {
  return useQuery<DatasetInfo[]>({
    queryKey: ['datasets'],
    queryFn: async () => {
      const result = await mcpCall<{ datasets: DatasetInfo[] }>('dataset.list')
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
