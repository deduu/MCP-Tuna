import { create } from 'zustand'
import type { ChatContentBlock } from '@/lib/chat-content'
import type { ReflectionEvent, ToolCallEvent } from '@/stores/chat'

export type CompareTargetKind = 'agent' | 'deployment'

export interface CompareMetrics {
  prompt_tokens?: number | null
  completion_tokens?: number | null
  total_tokens?: number | null
  latency_ms?: number | null
  ttft_ms?: number | null
  output_tokens_per_second?: number | null
  estimated_cost_usd?: number | null
  confidence?: string | null
  perplexity?: number | null
  tool_call_count?: number | null
  tool_time_ms?: number | null
  tool_names?: string[]
}

export interface CompareTargetConfig {
  id: string
  kind: CompareTargetKind
  label: string
  model?: string
  deploymentId?: string | null
  deploymentLabel?: string | null
  deploymentModality?: 'text' | 'vision-language' | 'unknown'
}

export interface CompareMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  parts?: ChatContentBlock[]
  thinking: string[]
  toolCalls: ToolCallEvent[]
  reflections: ReflectionEvent[]
  metrics?: CompareMetrics | null
  isStreaming?: boolean
  error?: string | null
  modelId?: string | null
}

export interface CompareSession {
  target: CompareTargetConfig
  messages: CompareMessage[]
  status: 'idle' | 'streaming' | 'error'
  conversationId: string | null
  abortController: AbortController | null
}

interface CompareStore {
  sessions: CompareSession[]
  baselineTargetId: string | null
  addTarget: (target: CompareTargetConfig) => void
  updateTarget: (targetId: string, patch: Partial<CompareTargetConfig>) => void
  removeTarget: (targetId: string) => void
  setBaselineTargetId: (targetId: string | null) => void
  addUserMessage: (targetId: string, message: { content: string; parts?: ChatContentBlock[] }) => void
  startAssistantMessage: (targetId: string) => string
  appendToken: (targetId: string, messageId: string, token: string) => void
  addThinking: (targetId: string, messageId: string, content: string) => void
  addToolStart: (targetId: string, messageId: string, tool: string, args: Record<string, unknown>) => void
  addToolEnd: (targetId: string, messageId: string, tool: string, durationMs: number) => void
  addReflection: (targetId: string, messageId: string, isReady: boolean, explanation: string) => void
  setMessageMetrics: (targetId: string, messageId: string, metrics: CompareMetrics) => void
  finishAssistantMessage: (targetId: string, messageId: string, patch?: Partial<CompareMessage>) => void
  failAssistantMessage: (targetId: string, messageId: string, error: string) => void
  setConversationId: (targetId: string, conversationId: string | null) => void
  setAbortController: (targetId: string, controller: AbortController | null) => void
  clearTargetMessages: (targetId: string) => void
  clearAllMessages: () => void
}

function genId(): string {
  return crypto.randomUUID()
}

function updateMessage(
  messages: CompareMessage[],
  messageId: string,
  updater: (message: CompareMessage) => CompareMessage,
) {
  return messages.map((message) => (message.id === messageId ? updater(message) : message))
}

export const useChatCompareStore = create<CompareStore>((set) => ({
  sessions: [],
  baselineTargetId: null,

  addTarget: (target) =>
    set((state) => {
      if (state.sessions.some((session) => session.target.id === target.id)) {
        return state
      }
      return {
        sessions: [
          ...state.sessions,
          {
            target,
            messages: [],
            status: 'idle',
            conversationId: null,
            abortController: null,
          },
        ],
        baselineTargetId: state.baselineTargetId ?? target.id,
      }
    }),

  updateTarget: (targetId, patch) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              target: { ...session.target, ...patch },
              conversationId:
                patch.kind && patch.kind !== session.target.kind
                  ? null
                  : session.conversationId,
              messages:
                patch.kind && patch.kind !== session.target.kind
                  ? []
                  : patch.model && patch.model !== session.target.model
                    ? []
                    : patch.deploymentId !== undefined && patch.deploymentId !== session.target.deploymentId
                      ? []
                      : session.messages,
            }
          : session,
      ),
    })),

  removeTarget: (targetId) =>
    set((state) => {
      const remaining = state.sessions.filter((session) => session.target.id !== targetId)
      return {
        sessions: remaining,
        baselineTargetId:
          state.baselineTargetId === targetId ? (remaining[0]?.target.id ?? null) : state.baselineTargetId,
      }
    }),

  setBaselineTargetId: (targetId) => set({ baselineTargetId: targetId }),

  addUserMessage: (targetId, message) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              messages: [
                ...session.messages,
                {
                  id: genId(),
                  role: 'user',
                  content: message.content,
                  parts: message.parts,
                  thinking: [],
                  toolCalls: [],
                  reflections: [],
                  metrics: null,
                },
              ],
            }
          : session,
      ),
    })),

  startAssistantMessage: (targetId) => {
    const id = genId()
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              status: 'streaming',
              messages: [
                ...session.messages,
                {
                  id,
                  role: 'assistant',
                  content: '',
                  thinking: [],
                  toolCalls: [],
                  reflections: [],
                  metrics: null,
                  isStreaming: true,
                },
              ],
            }
          : session,
      ),
    }))
    return id
  },

  appendToken: (targetId, messageId, token) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              messages: updateMessage(session.messages, messageId, (message) => ({
                ...message,
                content: message.content + token,
              })),
            }
          : session,
      ),
    })),

  addThinking: (targetId, messageId, content) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              messages: updateMessage(session.messages, messageId, (message) => ({
                ...message,
                thinking: [...message.thinking, content],
              })),
            }
          : session,
      ),
    })),

  addToolStart: (targetId, messageId, tool, args) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              messages: updateMessage(session.messages, messageId, (message) => ({
                ...message,
                toolCalls: [...message.toolCalls, { tool, arguments: args }],
              })),
            }
          : session,
      ),
    })),

  addToolEnd: (targetId, messageId, tool, durationMs) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              messages: updateMessage(session.messages, messageId, (message) => ({
                ...message,
                toolCalls: message.toolCalls.map((toolCall) =>
                  toolCall.tool === tool && toolCall.durationMs === undefined
                    ? { ...toolCall, durationMs }
                    : toolCall,
                ),
              })),
            }
          : session,
      ),
    })),

  addReflection: (targetId, messageId, isReady, explanation) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              messages: updateMessage(session.messages, messageId, (message) => ({
                ...message,
                reflections: [...message.reflections, { isReady, explanation }],
              })),
            }
          : session,
      ),
    })),

  setMessageMetrics: (targetId, messageId, metrics) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              messages: updateMessage(session.messages, messageId, (message) => ({
                ...message,
                metrics: { ...(message.metrics ?? {}), ...metrics },
              })),
            }
          : session,
      ),
    })),

  finishAssistantMessage: (targetId, messageId, patch) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              status: 'idle',
              abortController: null,
              messages: updateMessage(session.messages, messageId, (message) => ({
                ...message,
                ...patch,
                isStreaming: false,
                error: patch?.error ?? null,
              })),
            }
          : session,
      ),
    })),

  failAssistantMessage: (targetId, messageId, error) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? {
              ...session,
              status: 'error',
              abortController: null,
              messages: updateMessage(session.messages, messageId, (message) => ({
                ...message,
                content: message.content || `[Error: ${error}]`,
                isStreaming: false,
                error,
              })),
            }
          : session,
      ),
    })),

  setConversationId: (targetId, conversationId) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId ? { ...session, conversationId } : session,
      ),
    })),

  setAbortController: (targetId, controller) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? { ...session, abortController: controller, status: controller ? 'streaming' : session.status }
          : session,
      ),
    })),

  clearTargetMessages: (targetId) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.target.id === targetId
          ? { ...session, messages: [], conversationId: null, status: 'idle', abortController: null }
          : session,
      ),
    })),

  clearAllMessages: () =>
    set((state) => ({
      sessions: state.sessions.map((session) => ({
        ...session,
        messages: [],
        conversationId: null,
        status: 'idle',
        abortController: null,
      })),
    })),
}))
