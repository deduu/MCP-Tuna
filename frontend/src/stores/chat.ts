import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ChatContentBlock } from '@/lib/chat-content'

export interface ToolCallEvent {
  tool: string
  arguments: Record<string, unknown>
  durationMs?: number
}

export interface ReflectionEvent {
  isReady: boolean
  explanation: string
}

export interface TurnMetrics {
  turn: number
  confidence?: string | null
  perplexity?: number | null
  tokens?: number | null
  usage?: { prompt_tokens: number; completion_tokens: number } | null
}

export interface ConfirmationRequest {
  tool: string
  arguments: Record<string, unknown>
  message: string
}

export interface AgentEvent {
  type: 'thinking' | 'tool_start' | 'tool_end' | 'reflection' | 'phase' | 'metrics' | 'confirmation_needed'
  timestamp: number
  data: unknown
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  parts?: ChatContentBlock[]
  events: AgentEvent[]
  toolCalls: ToolCallEvent[]
  reflections: ReflectionEvent[]
  metrics: TurnMetrics[]
  thinking: string[]
  isStreaming?: boolean
  confirmation?: ConfirmationRequest
}

interface ChatStore {
  messages: ChatMessage[]
  isStreaming: boolean
  abortController: AbortController | null
  chatMode: 'agent' | 'deployment'
  selectedModel: string
  selectedDeploymentId: string | null
  deploymentConversationId: string | null

  addUserMessage: (message: { content: string; parts?: ChatContentBlock[] }) => void
  startAssistantMessage: () => string
  appendToken: (id: string, token: string) => void
  addThinking: (id: string, content: string) => void
  addToolStart: (id: string, tool: string, args: Record<string, unknown>) => void
  addToolEnd: (id: string, tool: string, durationMs: number) => void
  addReflection: (id: string, isReady: boolean, explanation: string) => void
  addMetrics: (id: string, metrics: TurnMetrics) => void
  addPhase: (id: string, phase: string, action: string) => void
  addConfirmation: (id: string, confirmation: ConfirmationRequest) => void
  resolveConfirmation: (id: string) => void
  finishAssistantMessage: (id: string, history?: unknown[]) => void
  setStreaming: (streaming: boolean, controller?: AbortController | null) => void
  setChatMode: (mode: 'agent' | 'deployment') => void
  setSelectedModel: (model: string) => void
  setSelectedDeploymentId: (deploymentId: string | null) => void
  setDeploymentConversationId: (conversationId: string | null) => void
  replaceMessages: (messages: ChatMessage[], conversationId?: string | null) => void
  clearMessages: () => void
}

function genId(): string {
  return crypto.randomUUID()
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      messages: [],
      isStreaming: false,
      abortController: null,
      chatMode: 'agent',
      selectedModel: 'gpt-4o',
      selectedDeploymentId: null,
      deploymentConversationId: null,

      addUserMessage: ({ content, parts }) =>
        set((s) => ({
          messages: [
            ...s.messages,
            {
              id: genId(),
              role: 'user',
              content,
              parts,
              events: [],
              toolCalls: [],
              reflections: [],
              metrics: [],
              thinking: [],
            },
          ],
        })),

      startAssistantMessage: () => {
        const id = genId()
        set((s) => ({
          messages: [
            ...s.messages,
            {
              id,
              role: 'assistant',
              content: '',
              events: [],
              toolCalls: [],
              reflections: [],
              metrics: [],
              thinking: [],
              isStreaming: true,
            },
          ],
        }))
        return id
      },

      appendToken: (id, token) =>
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === id ? { ...m, content: m.content + token } : m,
          ),
        })),

      addThinking: (id, content) =>
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === id
              ? {
                  ...m,
                  thinking: [...m.thinking, content],
                  events: [
                    ...m.events,
                    { type: 'thinking' as const, timestamp: Date.now(), data: content },
                  ],
                }
              : m,
          ),
        })),

      addToolStart: (id, tool, args) =>
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === id
              ? {
                  ...m,
                  toolCalls: [...m.toolCalls, { tool, arguments: args }],
                  events: [
                    ...m.events,
                    {
                      type: 'tool_start' as const,
                      timestamp: Date.now(),
                      data: { tool, arguments: args },
                    },
                  ],
                }
              : m,
          ),
        })),

      addToolEnd: (id, tool, durationMs) =>
        set((s) => ({
          messages: s.messages.map((m) => {
            if (m.id !== id) return m
            const toolCalls = m.toolCalls.map((tc) =>
              tc.tool === tool && tc.durationMs === undefined
                ? { ...tc, durationMs }
                : tc,
            )
            return {
              ...m,
              toolCalls,
              events: [
                ...m.events,
                {
                  type: 'tool_end' as const,
                  timestamp: Date.now(),
                  data: { tool, durationMs },
                },
              ],
            }
          }),
        })),

      addReflection: (id, isReady, explanation) =>
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === id
              ? {
                  ...m,
                  reflections: [...m.reflections, { isReady, explanation }],
                  events: [
                    ...m.events,
                    {
                      type: 'reflection' as const,
                      timestamp: Date.now(),
                      data: { isReady, explanation },
                    },
                  ],
                }
              : m,
          ),
        })),

      addMetrics: (id, metrics) =>
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === id
              ? {
                  ...m,
                  metrics: [...m.metrics, metrics],
                  events: [
                    ...m.events,
                    { type: 'metrics' as const, timestamp: Date.now(), data: metrics },
                  ],
                }
              : m,
          ),
        })),

      addPhase: (id, phase, action) =>
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === id
              ? {
                  ...m,
                  events: [
                    ...m.events,
                    {
                      type: 'phase' as const,
                      timestamp: Date.now(),
                      data: { phase, action },
                    },
                  ],
                }
              : m,
          ),
        })),

      addConfirmation: (id, confirmation) =>
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === id
              ? {
                  ...m,
                  confirmation,
                  events: [
                    ...m.events,
                    {
                      type: 'confirmation_needed' as const,
                      timestamp: Date.now(),
                      data: confirmation,
                    },
                  ],
                }
              : m,
          ),
        })),

      resolveConfirmation: (id) =>
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === id ? { ...m, confirmation: undefined } : m,
          ),
        })),

      finishAssistantMessage: (id, history) =>
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === id
              ? {
                  ...m,
                  isStreaming: false,
                  thinking:
                    m.thinking.length === 0 && Array.isArray(history)
                      ? history
                          .filter((h): h is Record<string, unknown> => h != null && typeof h === 'object' && 'thought' in (h as Record<string, unknown>) && Boolean((h as Record<string, unknown>).thought))
                          .map((h) => (h as Record<string, unknown>).thought as string)
                      : m.thinking,
                }
              : m,
          ),
        })),

      setStreaming: (streaming, controller) =>
        set({ isStreaming: streaming, abortController: controller ?? null }),

      setChatMode: (chatMode) =>
        set((state) =>
          state.chatMode === chatMode
            ? state
            : { chatMode, deploymentConversationId: null, messages: [] },
        ),

      setSelectedModel: (selectedModel) =>
        set((state) =>
          state.selectedModel === selectedModel
            ? state
            : { selectedModel, deploymentConversationId: null, messages: [] },
        ),

      setSelectedDeploymentId: (selectedDeploymentId) =>
        set((state) =>
          state.selectedDeploymentId === selectedDeploymentId
            ? state
            : { selectedDeploymentId, deploymentConversationId: null, messages: [] },
        ),

      setDeploymentConversationId: (deploymentConversationId) => set({ deploymentConversationId }),

      replaceMessages: (messages, conversationId) =>
        set({
          messages: messages.map((message) => ({ ...message, isStreaming: false })),
          deploymentConversationId: conversationId ?? null,
        }),

      clearMessages: () => set({ messages: [], deploymentConversationId: null }),
    }),
    {
      name: 'mcp-tuna-chat',
      partialize: (state) => ({
        messages: state.messages.map((message) => ({ ...message, isStreaming: false })),
        chatMode: state.chatMode,
        selectedModel: state.selectedModel,
        selectedDeploymentId: state.selectedDeploymentId,
        deploymentConversationId: state.deploymentConversationId,
      }),
    },
  ),
)
