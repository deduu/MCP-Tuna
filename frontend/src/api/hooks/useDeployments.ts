import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mcpCall } from '../client'
import type { Deployment } from '../types'

export function useDeployments() {
  return useQuery<Deployment[]>({
    queryKey: ['deployments'],
    queryFn: async () => {
      const result = await mcpCall<{ deployments: Deployment[] }>('host.list_deployments')
      return result.deployments ?? []
    },
    refetchInterval: 10_000,
    retry: 1,
  })
}

export function useDeploymentStatus(deploymentId: string) {
  return useQuery({
    queryKey: ['deployments', 'status', deploymentId],
    queryFn: () => mcpCall<Record<string, unknown>>('host.get_status', { deployment_id: deploymentId }),
    enabled: !!deploymentId,
    refetchInterval: 5_000,
  })
}

export function useDeploymentLogs(deploymentId: string, enabled: boolean) {
  return useQuery<string[]>({
    queryKey: ['deployments', 'logs', deploymentId],
    queryFn: async () => {
      const result = await mcpCall<{ logs: string[] }>('host.get_logs', { deployment_id: deploymentId })
      return result.logs ?? []
    },
    enabled: !!deploymentId && enabled,
    refetchInterval: 3_000,
  })
}

type DeployParams = {
  type: 'mcp' | 'api'
  args: Record<string, unknown>
}

export function useDeploy() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ type, args }: DeployParams) =>
      mcpCall(type === 'mcp' ? 'host.deploy_mcp' : 'host.deploy_api', args),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['deployments'] })
    },
  })
}

export function useStopDeployment() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (deploymentId: string) =>
      mcpCall('host.stop', { deployment_id: deploymentId }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['deployments'] })
    },
  })
}

export function useUndeployment() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (deploymentId: string) =>
      mcpCall('host.undeploy', { deployment_id: deploymentId }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['deployments'] })
    },
  })
}
