import type { ChatContentBlock, ChatImageBlock } from '@/lib/chat-content'
import { sanitizeChatContentForRequest } from '@/lib/chat-content'

type MessageBlock = Record<string, unknown>
type MessageLike = {
  role?: unknown
  content?: unknown
}

function normalizeContentBlocks(content: unknown): MessageBlock[] {
  if (typeof content === 'string') {
    return [{ type: 'text', text: content }]
  }
  if (!Array.isArray(content)) {
    return []
  }

  return content.filter((item): item is MessageBlock => !!item && typeof item === 'object')
}

function hasImageBlock(content: unknown): boolean {
  return normalizeContentBlocks(content).some((block) => {
    const type = block.type
    return type === 'image_path' || type === 'image_url' || type === 'image_base64'
  })
}

export function buildEvaluationMessages(
  prompt: string,
  images: ChatImageBlock[],
): Array<Record<string, unknown>> {
  const content: ChatContentBlock[] = [
    ...images,
    ...(prompt.trim() ? [{ type: 'text' as const, text: prompt.trim() }] : []),
  ]
  return [
    {
      role: 'user',
      content: sanitizeChatContentForRequest(images.length > 0 ? content : prompt.trim()),
    },
  ]
}

export function isVlmDatasetRow(row: unknown): row is { messages: MessageLike[] } {
  if (!row || typeof row !== 'object' || !Array.isArray((row as { messages?: unknown }).messages)) {
    return false
  }
  return (row as { messages: MessageLike[] }).messages.some((message) => hasImageBlock(message.content))
}

export function extractAssistantText(messages: unknown): string {
  if (!Array.isArray(messages)) {
    return ''
  }

  return messages
    .filter((message): message is MessageLike => !!message && typeof message === 'object')
    .filter((message) => String(message.role ?? '').toLowerCase() === 'assistant')
    .flatMap((message) => normalizeContentBlocks(message.content))
    .filter((block) => block.type === 'text' && typeof block.text === 'string')
    .map((block) => String(block.text).trim())
    .filter(Boolean)
    .join('\n')
}

export function extractPromptText(messages: unknown): string {
  if (!Array.isArray(messages)) {
    return ''
  }

  return messages
    .filter((message): message is MessageLike => !!message && typeof message === 'object')
    .filter((message) => {
      const role = String(message.role ?? '').toLowerCase()
      return role === 'user' || role === 'system'
    })
    .flatMap((message) => normalizeContentBlocks(message.content))
    .filter((block) => block.type === 'text' && typeof block.text === 'string')
    .map((block) => String(block.text).trim())
    .filter(Boolean)
    .join('\n')
}
