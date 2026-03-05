import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mcpCall } from '../client'
import type { SystemResources, SetupCheckResult } from '../types'

export function useSystemResources() {
  return useQuery<SystemResources>({
    queryKey: ['system', 'resources'],
    queryFn: () => mcpCall<SystemResources>('system.check_resources'),
    refetchInterval: 30_000,
    retry: 1,
  })
}

export function useSetupCheck() {
  return useQuery<SetupCheckResult>({
    queryKey: ['system', 'setup'],
    queryFn: () => mcpCall<SetupCheckResult>('system.setup_check'),
    staleTime: 60_000,
    retry: 1,
  })
}

export function useSetHFToken() {
  const qc = useQueryClient()
  return useMutation<{ success: boolean; username?: string; warning?: string }, Error, string>({
    mutationFn: (token: string) => mcpCall('system.set_hf_token', { token }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['system', 'setup'] })
    },
  })
}
