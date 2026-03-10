import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mcpCall } from '../client'
import type {
  SystemResources,
  SetupCheckResult,
  GatewayConfigResult,
  SystemHealthResult,
} from '../types'

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

export function useSystemHealth() {
  return useQuery<SystemHealthResult>({
    queryKey: ['system', 'health'],
    queryFn: () => mcpCall<SystemHealthResult>('system.health'),
    refetchInterval: 30_000,
    retry: 1,
  })
}

export function useSystemConfig() {
  return useQuery<GatewayConfigResult>({
    queryKey: ['system', 'config'],
    queryFn: () => mcpCall<GatewayConfigResult>('system.config'),
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
      qc.invalidateQueries({ queryKey: ['system', 'config'] })
    },
  })
}

export function useSetRuntimeEnv() {
  const qc = useQueryClient()
  return useMutation<
    { success: boolean; key: string; configured: boolean; message: string },
    Error,
    { key: string; value?: string }
  >({
    mutationFn: ({ key, value }) => mcpCall('system.set_runtime_env', { key, value }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['system', 'setup'] })
      qc.invalidateQueries({ queryKey: ['system', 'config'] })
    },
  })
}
