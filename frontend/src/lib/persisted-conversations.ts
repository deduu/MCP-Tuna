import type { ConversationMessage } from '@/api/types'
import type { ChatMessage } from '@/stores/chat'
import type { ChatContentBlock } from '@/lib/chat-content'
import { extractTextFromChatContent } from '@/lib/chat-content'

export function persistedConversationToChatMessages(
  messages: ConversationMessage[],
): ChatMessage[] {
  return messages.map((message) => {
    const parts = parseStructuredContent(message.content)
    return {
      id: `persisted-${message.sequence}`,
      role: message.role,
      content: parts
        ? extractTextFromChatContent(parts)
        : typeof message.content === 'string'
          ? message.content
          : '',
      parts,
      events: [],
      toolCalls: [],
      reflections: [],
      metrics: [],
      thinking: [],
      isStreaming: false,
    }
  })
}

function parseStructuredContent(
  content: ConversationMessage['content'],
): ChatContentBlock[] | undefined {
  if (!Array.isArray(content)) {
    return undefined
  }

  const blocks: ChatContentBlock[] = []
  for (const item of content) {
    if (typeof item !== 'object' || item === null) {
      continue
    }
    if (item.type === 'text' && typeof item.text === 'string') {
      blocks.push({ type: 'text', text: item.text })
      continue
    }
    if (item.type === 'image_path' && typeof item.image_path === 'string') {
      blocks.push({ type: 'image_path', image_path: item.image_path })
    }
  }

  return blocks.length > 0 ? blocks : undefined
}
