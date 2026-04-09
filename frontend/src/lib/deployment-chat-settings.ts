import type { ThinkingMode } from '@/api/types'

const DEPLOYMENT_CHAT_SETTINGS_KEY_PREFIX = 'agentsoul.deploymentChatSettings.'

export interface DeploymentChatDraftSettings {
  systemPrompt: string
  thinkingMode: ThinkingMode
  temperature: string
  topP: string
  topK: string
  maxNewTokens: string
}

export const DEFAULT_DEPLOYMENT_CHAT_SETTINGS: DeploymentChatDraftSettings = {
  systemPrompt: '',
  thinkingMode: 'default',
  temperature: '0.7',
  topP: '0.95',
  topK: '50',
  maxNewTokens: '512',
}

function storageKey(deploymentId: string) {
  return `${DEPLOYMENT_CHAT_SETTINGS_KEY_PREFIX}${deploymentId}`
}

function readString(value: unknown) {
  return typeof value === 'string' ? value : undefined
}

function readThinkingMode(value: unknown): ThinkingMode | undefined {
  return value === 'default' || value === 'on' || value === 'off'
    ? value
    : undefined
}

export function getDeploymentChatDraftSettings(
  deploymentId: string,
): Partial<DeploymentChatDraftSettings> | null {
  if (typeof window === 'undefined' || !deploymentId.trim()) {
    return null
  }

  try {
    const raw = window.localStorage.getItem(storageKey(deploymentId))
    if (!raw) {
      return null
    }

    const parsed = JSON.parse(raw) as Record<string, unknown>
    return {
      systemPrompt: readString(parsed.systemPrompt),
      thinkingMode: readThinkingMode(parsed.thinkingMode),
      temperature: readString(parsed.temperature),
      topP: readString(parsed.topP),
      topK: readString(parsed.topK),
      maxNewTokens: readString(parsed.maxNewTokens),
    }
  } catch {
    return null
  }
}

export function setDeploymentChatDraftSettings(
  deploymentId: string,
  settings: DeploymentChatDraftSettings,
) {
  if (typeof window === 'undefined' || !deploymentId.trim()) {
    return
  }

  window.localStorage.setItem(storageKey(deploymentId), JSON.stringify(settings))
}
