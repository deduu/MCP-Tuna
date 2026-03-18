export interface ChatTextBlock {
  type: 'text'
  text: string
}

export interface ChatImageBlock {
  type: 'image_path'
  image_path: string
  preview_url?: string
  file_name?: string
}

export type ChatContentBlock = ChatTextBlock | ChatImageBlock

export function isStructuredChatContent(
  content: string | ChatContentBlock[] | undefined,
): content is ChatContentBlock[] {
  return Array.isArray(content)
}

export function extractTextFromChatContent(
  content: string | ChatContentBlock[] | undefined,
): string {
  if (typeof content === 'string') {
    return content
  }

  if (!Array.isArray(content)) {
    return ''
  }

  return content
    .filter((block): block is ChatTextBlock => block.type === 'text')
    .map((block) => block.text.trim())
    .filter(Boolean)
    .join('\n')
}

export function sanitizeChatContentForRequest(
  content: string | ChatContentBlock[],
): string | Array<Record<string, unknown>> {
  if (typeof content === 'string') {
    return content
  }

  return content.map((block) => {
    if (block.type === 'text') {
      return { type: 'text', text: block.text }
    }

    return { type: 'image_path', image_path: block.image_path }
  })
}

export function buildUserChatContent(
  text: string,
  imageBlocks: ChatImageBlock[],
): string | ChatContentBlock[] {
  const trimmed = text.trim()
  if (imageBlocks.length === 0) {
    return trimmed
  }

  const blocks: ChatContentBlock[] = [...imageBlocks]
  if (trimmed) {
    blocks.push({ type: 'text', text: trimmed })
  }
  return blocks
}
