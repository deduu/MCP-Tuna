import { mcpCall } from '@/api/client'
import {
  extractTextFromChatContent,
  sanitizeChatContentForRequest,
  type ChatContentBlock,
} from '@/lib/chat-content'
import type { CompareJudgement } from '@/stores/chatCompare'

interface JudgeAgainstBaselineParams {
  promptText: string
  promptParts?: ChatContentBlock[]
  baselineResponse: string
  targetResponse: string
}

function hasImageBlocks(parts: ChatContentBlock[] | undefined): boolean {
  return Array.isArray(parts) && parts.some((block) => block.type === 'image_path')
}

function normalizeWinner(winner: unknown): 'baseline' | 'target' | 'tie' {
  const raw = String(winner ?? '').trim().toUpperCase()
  if (raw === 'A') return 'baseline'
  if (raw === 'B') return 'target'
  return 'tie'
}

export async function judgeAgainstBaseline({
  promptText,
  promptParts,
  baselineResponse,
  targetResponse,
}: JudgeAgainstBaselineParams): Promise<CompareJudgement> {
  const textPrompt = promptText.trim() || extractTextFromChatContent(promptParts)
  if (!textPrompt && !hasImageBlocks(promptParts)) {
    throw new Error('A prompt is required before judging responses')
  }

  const toolName = hasImageBlocks(promptParts) ? 'judge.compare_vlm' : 'judge.evaluate'
  const payload = await mcpCall<Record<string, unknown>>(
    toolName,
    hasImageBlocks(promptParts)
      ? {
          messages: [
            {
              role: 'user',
              content: sanitizeChatContentForRequest(promptParts ?? []),
            },
          ],
          generated_a: baselineResponse,
          generated_b: targetResponse,
        }
      : {
          question: textPrompt,
          generated: baselineResponse,
          generated_b: targetResponse,
          judge_type: 'pairwise',
        },
  )

  const rawResult = payload.result && typeof payload.result === 'object'
    ? (payload.result as Record<string, unknown>)
    : payload

  return {
    winner: normalizeWinner(rawResult.winner),
    confidence: typeof rawResult.confidence === 'number' ? rawResult.confidence : null,
    rationale:
      typeof rawResult.reason === 'string'
        ? rawResult.reason
        : typeof rawResult.explanation === 'string'
          ? rawResult.explanation
          : null,
    toolName,
    judgedAt: new Date().toISOString(),
  }
}
