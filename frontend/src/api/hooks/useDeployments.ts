import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mcpCall } from '../client'
import type {
  Deployment,
  DeploymentConversation,
  DeploymentConversationSummary,
  ModelModality,
} from '../types'

function normalizeDeployment(raw: Record<string, unknown>): Deployment {
  const deploymentId =
    typeof raw.deployment_id === 'string'
      ? raw.deployment_id
      : typeof raw.id === 'string'
        ? raw.id
        : ''

  const type =
    raw.type === 'api' || raw.type === 'mcp'
      ? raw.type
      : raw.transport === 'stdio'
        ? 'mcp'
        : 'mcp'

  const status = raw.status === 'stopped' ? 'stopped' : 'running'

  return {
    deployment_id: deploymentId,
    name: typeof raw.name === 'string' ? raw.name : undefined,
    system_prompt: typeof raw.system_prompt === 'string' ? raw.system_prompt : null,
    model_path: typeof raw.model_path === 'string' ? raw.model_path : '',
    adapter_path: typeof raw.adapter_path === 'string' ? raw.adapter_path : undefined,
    endpoint: typeof raw.endpoint === 'string' ? raw.endpoint : '',
    type,
    status,
    transport: typeof raw.transport === 'string' ? raw.transport : undefined,
    modality:
      raw.modality === 'vision-language' || raw.modality === 'unknown'
        ? raw.modality
        : 'text',
    routes: Array.isArray(raw.routes)
      ? raw.routes.filter((route): route is string => typeof route === 'string')
      : undefined,
    created_at: typeof raw.created_at === 'string' ? raw.created_at : undefined,
    updated_at: typeof raw.updated_at === 'string' ? raw.updated_at : undefined,
    stopped_at: typeof raw.stopped_at === 'string' ? raw.stopped_at : undefined,
  }
}

function normalizeConversationSummary(raw: Record<string, unknown>): DeploymentConversationSummary {
  return {
    conversation_id: typeof raw.conversation_id === 'string' ? raw.conversation_id : '',
    title: typeof raw.title === 'string' ? raw.title : null,
    deployment_id: typeof raw.deployment_id === 'string' ? raw.deployment_id : null,
    modality:
      raw.modality === 'vision-language' || raw.modality === 'unknown'
        ? raw.modality
        : 'text',
    endpoint: typeof raw.endpoint === 'string' ? raw.endpoint : null,
    model_path: typeof raw.model_path === 'string' ? raw.model_path : null,
    adapter_path: typeof raw.adapter_path === 'string' ? raw.adapter_path : null,
    message_count: typeof raw.message_count === 'number' ? raw.message_count : 0,
    created_at: typeof raw.created_at === 'string' ? raw.created_at : undefined,
    updated_at: typeof raw.updated_at === 'string' ? raw.updated_at : undefined,
  }
}

function normalizeConversation(raw: Record<string, unknown>): DeploymentConversation {
  const summary = normalizeConversationSummary(raw)
  return {
    ...summary,
    system_prompt: typeof raw.system_prompt === 'string' ? raw.system_prompt : null,
    messages: Array.isArray(raw.messages)
      ? raw.messages
          .filter((message): message is Record<string, unknown> => typeof message === 'object' && message !== null)
          .map((message) => ({
            sequence: typeof message.sequence === 'number' ? message.sequence : 0,
            role: message.role === 'assistant' ? 'assistant' : 'user',
            content:
              typeof message.content === 'string' || Array.isArray(message.content)
                ? message.content
                : '',
          }))
      : [],
  }
}

function sortTimestamp(value?: string) {
  const parsed = Date.parse(value ?? '')
  return Number.isFinite(parsed) ? parsed : 0
}

export function useDeployments() {
  return useQuery<Deployment[]>({
    queryKey: ['deployments'],
    queryFn: async () => {
      const result = await mcpCall<{ deployments: Array<Record<string, unknown>> }>('host.list_deployments')
      return (result.deployments ?? [])
        .map(normalizeDeployment)
        .sort((a, b) => {
          if (a.status !== b.status) {
            return a.status === 'running' ? -1 : 1
          }
          return sortTimestamp(b.updated_at ?? b.created_at) - sortTimestamp(a.updated_at ?? a.created_at)
        })
    },
    refetchInterval: 10_000,
    retry: 1,
  })
}

export function useDeploymentStatus(deploymentId: string) {
  return useQuery({
    queryKey: ['deployments', 'status', deploymentId],
    queryFn: () => mcpCall<Record<string, unknown>>('host.health', { deployment_id: deploymentId }),
    enabled: !!deploymentId,
    refetchInterval: 5_000,
  })
}

export function useDeploymentLogs(deploymentId: string, enabled: boolean) {
  return useQuery<string[]>({
    queryKey: ['deployments', 'logs', deploymentId],
    queryFn: async () => {
      const result = await mcpCall<Record<string, unknown>>('host.health', { deployment_id: deploymentId })
      return [JSON.stringify(result, null, 2)]
    },
    enabled: !!deploymentId && enabled,
    refetchInterval: 3_000,
  })
}

export function useDeploymentConversations(deploymentId: string, enabled: boolean = true) {
  return useQuery<DeploymentConversationSummary[]>({
    queryKey: ['deployments', 'conversations', deploymentId],
    queryFn: async () => {
      const result = await mcpCall<{ conversations: Array<Record<string, unknown>> }>(
        'host.list_conversations',
        { deployment_id: deploymentId, limit: 25 },
      )
      return (result.conversations ?? []).map(normalizeConversationSummary)
    },
    enabled: enabled && !!deploymentId,
    refetchInterval: 10_000,
    retry: 1,
  })
}

export function useDeploymentConversation(conversationId: string | null, enabled: boolean = true) {
  return useQuery<DeploymentConversation>({
    queryKey: ['deployments', 'conversation', conversationId],
    queryFn: async () =>
      normalizeConversation(
        await mcpCall<Record<string, unknown>>('host.get_conversation', {
          conversation_id: conversationId,
        }),
      ),
    enabled: enabled && !!conversationId,
    staleTime: 30_000,
    retry: 1,
  })
}

export function useRenameConversation() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ conversationId, title }: { conversationId: string; title: string }) =>
      mcpCall('host.rename_conversation', { conversation_id: conversationId, title }),
    onSuccess: (_result, variables) => {
      qc.invalidateQueries({ queryKey: ['deployments', 'conversation', variables.conversationId] })
      qc.invalidateQueries({ queryKey: ['deployments', 'conversations'] })
    },
  })
}

export function useDeleteConversation() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (conversationId: string) =>
      mcpCall('host.delete_conversation', { conversation_id: conversationId }),
    onSuccess: (_result, conversationId) => {
      qc.removeQueries({ queryKey: ['deployments', 'conversation', conversationId] })
      qc.invalidateQueries({ queryKey: ['deployments', 'conversations'] })
    },
  })
}

type DeployParams = {
  type: 'mcp' | 'api'
  modality?: ModelModality
  args: Record<string, unknown>
}

const DEPLOY_TIMEOUT_MS = 120_000

export function useDeploy() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ type, modality, args }: DeployParams) => {
      const resolvedModality = modality === 'vision-language' ? 'vision-language' : 'text'
      const toolName =
        type === 'mcp'
          ? resolvedModality === 'vision-language'
            ? 'host.deploy_vlm_mcp'
            : 'host.deploy_mcp'
          : resolvedModality === 'vision-language'
            ? 'host.deploy_vlm_api'
            : 'host.deploy_api'
      return mcpCall(toolName, args, { timeoutMs: DEPLOY_TIMEOUT_MS })
    },
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
      mcpCall('host.delete_deployment', { deployment_id: deploymentId }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['deployments'] })
    },
  })
}

export function getRedeployInitialValues(deployment: Deployment) {
  return {
    name: deployment.name,
    systemPrompt: deployment.system_prompt,
    modelPath: deployment.model_path,
    adapterPath: deployment.adapter_path,
    modality: deployment.modality === 'vision-language' ? 'vision-language' : 'text',
  } as const
}
