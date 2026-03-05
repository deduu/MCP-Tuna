import { useQuery } from '@tanstack/react-query'
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
