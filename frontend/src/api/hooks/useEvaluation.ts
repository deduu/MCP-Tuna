import { useQuery } from '@tanstack/react-query'
import { mcpCall } from '../client'

export interface JudgeCriterion {
  name: string
  description: string
  weight?: number
}

export function useJudgeConfig() {
  return useQuery<Record<string, unknown>>({
    queryKey: ['judge', 'config'],
    queryFn: () => mcpCall<Record<string, unknown>>('judge.get_config'),
    staleTime: 60_000,
    retry: 1,
  })
}

export function useJudgeCriteria() {
  return useQuery<JudgeCriterion[]>({
    queryKey: ['judge', 'criteria'],
    queryFn: async () => {
      const result = await mcpCall<{ criteria: JudgeCriterion[] }>('judge.list_criteria')
      return result.criteria ?? []
    },
    staleTime: 60_000,
    retry: 1,
  })
}
