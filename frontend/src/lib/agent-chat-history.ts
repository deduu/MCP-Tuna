import type { ChatContentBlock } from '@/lib/chat-content'
import type { ChatMessage } from '@/stores/chat'

const AGENT_CHAT_HISTORY_STORAGE_KEY = 'mcp-tuna-agent-chat-history'

interface StoredAgentConversationMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  parts?: ChatContentBlock[]
  toolCalls: ChatMessage['toolCalls']
  reflections: ChatMessage['reflections']
  metrics: ChatMessage['metrics']
  thinking: ChatMessage['thinking']
}

interface StoredAgentConversation {
  conversation_id: string
  title?: string | null
  model: string
  message_count: number
  created_at: string
  updated_at: string
  messages: StoredAgentConversationMessage[]
}

export interface AgentConversationSummary {
  conversation_id: string
  title?: string | null
  model: string
  message_count: number
  created_at: string
  updated_at: string
}

export interface AgentConversation extends AgentConversationSummary {
  messages: ChatMessage[]
}

export function listAgentConversations(): AgentConversationSummary[] {
  return readStoredConversations()
    .map(toSummary)
    .sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at))
}

export function getAgentConversation(conversationId: string): AgentConversation | null {
  const record = readStoredConversations().find(
    (conversation) => conversation.conversation_id === conversationId,
  )
  if (!record) {
    return null
  }

  return {
    ...toSummary(record),
    messages: restoreMessages(record.messages),
  }
}

export function upsertAgentConversation({
  conversationId,
  model,
  messages,
}: {
  conversationId?: string | null
  model: string
  messages: ChatMessage[]
}): AgentConversationSummary {
  const storedConversations = readStoredConversations()
  const safeMessages = storeMessages(messages)
  const now = new Date().toISOString()
  const existingConversation = conversationId
    ? storedConversations.find((conversation) => conversation.conversation_id === conversationId)
    : undefined
  const nextConversationId = existingConversation?.conversation_id ?? crypto.randomUUID()
  const hasConversationChanged =
    !existingConversation ||
    existingConversation.model !== model ||
    JSON.stringify(existingConversation.messages) !== JSON.stringify(safeMessages)

  if (existingConversation && !hasConversationChanged) {
    return toSummary(existingConversation)
  }

  const nextRecord: StoredAgentConversation = {
    conversation_id: nextConversationId,
    title: existingConversation?.title ?? inferConversationTitle(safeMessages, model),
    model,
    message_count: safeMessages.length,
    created_at: existingConversation?.created_at ?? now,
    updated_at: now,
    messages: safeMessages,
  }

  const nextConversations = [
    nextRecord,
    ...storedConversations.filter((conversation) => conversation.conversation_id !== nextConversationId),
  ]
  writeStoredConversations(nextConversations)
  return toSummary(nextRecord)
}

export function renameAgentConversation(conversationId: string, title: string) {
  const storedConversations = readStoredConversations()
  const nextConversations = storedConversations.map((conversation) =>
    conversation.conversation_id === conversationId
      ? { ...conversation, title, updated_at: new Date().toISOString() }
      : conversation,
  )
  writeStoredConversations(nextConversations)
}

export function deleteAgentConversation(conversationId: string) {
  const storedConversations = readStoredConversations().filter(
    (conversation) => conversation.conversation_id !== conversationId,
  )
  writeStoredConversations(storedConversations)
}

function toSummary(conversation: StoredAgentConversation): AgentConversationSummary {
  return {
    conversation_id: conversation.conversation_id,
    title: conversation.title,
    model: conversation.model,
    message_count: conversation.message_count,
    created_at: conversation.created_at,
    updated_at: conversation.updated_at,
  }
}

function readStoredConversations(): StoredAgentConversation[] {
  if (typeof window === 'undefined') {
    return []
  }

  const raw = window.localStorage.getItem(AGENT_CHAT_HISTORY_STORAGE_KEY)
  if (!raw) {
    return []
  }

  try {
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed.filter(isStoredConversation)
  } catch {
    return []
  }
}

function writeStoredConversations(conversations: StoredAgentConversation[]) {
  if (typeof window === 'undefined') {
    return
  }

  window.localStorage.setItem(
    AGENT_CHAT_HISTORY_STORAGE_KEY,
    JSON.stringify(conversations.slice(0, 40)),
  )
}

function storeMessages(messages: ChatMessage[]): StoredAgentConversationMessage[] {
  return messages
    .filter((message) => !message.isStreaming)
    .map((message) => ({
      id: message.id,
      role: message.role,
      content: message.content,
      parts: sanitizeParts(message.parts),
      toolCalls: message.toolCalls,
      reflections: message.reflections,
      metrics: message.metrics,
      thinking: message.thinking,
    }))
}

function restoreMessages(messages: StoredAgentConversationMessage[]): ChatMessage[] {
  return messages.map((message) => ({
    ...message,
    parts: sanitizeParts(message.parts),
    events: [],
    confirmation: undefined,
    isStreaming: false,
  }))
}

function sanitizeParts(parts?: ChatContentBlock[]): ChatContentBlock[] | undefined {
  if (!parts?.length) {
    return undefined
  }

  return parts.map((part) =>
    part.type === 'text'
      ? { type: 'text', text: part.text }
      : {
          type: 'image_path',
          image_path: part.image_path,
          file_name: part.file_name,
        },
  )
}

function inferConversationTitle(
  messages: StoredAgentConversationMessage[],
  model: string,
): string {
  const firstUserMessage = messages.find((message) => message.role === 'user')
  const userText =
    firstUserMessage?.parts
      ?.filter((part): part is Extract<ChatContentBlock, { type: 'text' }> => part.type === 'text')
      .map((part) => part.text.trim())
      .find(Boolean) ??
    firstUserMessage?.content.trim()

  if (userText) {
    return userText.length > 56 ? `${userText.slice(0, 53)}...` : userText
  }

  return `${model} session`
}

function isStoredConversation(value: unknown): value is StoredAgentConversation {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const candidate = value as Partial<StoredAgentConversation>
  return (
    typeof candidate.conversation_id === 'string' &&
    typeof candidate.model === 'string' &&
    typeof candidate.message_count === 'number' &&
    typeof candidate.created_at === 'string' &&
    typeof candidate.updated_at === 'string' &&
    Array.isArray(candidate.messages)
  )
}
