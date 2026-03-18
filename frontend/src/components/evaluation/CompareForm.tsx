import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import type { ChatImageBlock } from '@/lib/chat-content'
import { buildEvaluationMessages } from '@/lib/evaluation-multimodal'
import { toast } from 'sonner'
import { ImageAttachmentField } from './ImageAttachmentField'

interface CompareResult {
  winner?: string
  explanation?: string
  confidence?: number
  score_a?: number
  score_b?: number
  [key: string]: unknown
}

function normalizeCompareResult(payload: Record<string, unknown>): CompareResult {
  const raw = payload.result && typeof payload.result === 'object'
    ? (payload.result as Record<string, unknown>)
    : payload.aggregated && typeof payload.aggregated === 'object'
      ? (payload.aggregated as Record<string, unknown>)
      : payload

  return {
    winner: typeof raw.winner === 'string' ? raw.winner : undefined,
    explanation: typeof raw.explanation === 'string'
      ? raw.explanation
      : typeof raw.reason === 'string'
        ? raw.reason
        : undefined,
    confidence: typeof raw.confidence === 'number' ? raw.confidence : undefined,
    score_a: typeof raw.score_a === 'number' ? raw.score_a : undefined,
    score_b: typeof raw.score_b === 'number' ? raw.score_b : undefined,
  }
}

export function CompareForm() {
  const [question, setQuestion] = useState('')
  const [reference, setReference] = useState('')
  const [responseA, setResponseA] = useState('')
  const [responseB, setResponseB] = useState('')
  const [images, setImages] = useState<ChatImageBlock[]>([])
  const [isUploadingImages, setIsUploadingImages] = useState(false)
  const [result, setResult] = useState<CompareResult | null>(null)
  const { mutateAsync: executeTool, isPending } = useToolExecution()

  async function handleCompare() {
    if ((!question.trim() && images.length === 0) || !responseA.trim() || !responseB.trim()) {
      toast.error('A prompt or image plus both responses is required')
      return
    }
    try {
      const res = await executeTool({
        toolName: images.length > 0 ? 'judge.compare_vlm' : 'judge.compare_pair',
        args: images.length > 0
          ? {
              messages: buildEvaluationMessages(question, images),
              generated_a: responseA,
              generated_b: responseB,
              ...(reference.trim() ? { reference: reference.trim() } : {}),
            }
          : {
              question: question.trim(),
              generated_a: responseA,
              generated_b: responseB,
              ...(reference.trim() ? { reference: reference.trim() } : {}),
            },
      })
      setResult(normalizeCompareResult(res as Record<string, unknown>))
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Comparison failed')
    }
  }

  function winnerBadge(winner: string) {
    const normalized = winner.toLowerCase()
    if (normalized === 'a') return <Badge variant="success">Response A Wins</Badge>
    if (normalized === 'b') return <Badge variant="warning">Response B Wins</Badge>
    return <Badge variant="secondary">Tie</Badge>
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="pt-6 space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">Input / Prompt</label>
            <textarea
              placeholder={images.length > 0 ? 'Prompt text to pair with the attached images...' : 'Shared context or prompt...'}
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-20 resize-y"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>
          <ImageAttachmentField
            images={images}
            onChange={setImages}
            isUploading={isUploadingImages}
            onUploadingChange={setIsUploadingImages}
          />
          <div className="space-y-1">
            <label className="text-sm font-medium">Reference (optional)</label>
            <textarea
              placeholder="Reference answer..."
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-20 resize-y"
              value={reference}
              onChange={(e) => setReference(e.target.value)}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-sm font-medium">Response A</label>
              <textarea
                placeholder="First response..."
                className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-32 resize-y"
                value={responseA}
                onChange={(e) => setResponseA(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-medium">Response B</label>
              <textarea
                placeholder="Second response..."
                className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-32 resize-y"
                value={responseB}
                onChange={(e) => setResponseB(e.target.value)}
              />
            </div>
          </div>
          <Button onClick={handleCompare} disabled={isPending || isUploadingImages}>
            {isPending ? 'Comparing...' : images.length > 0 ? 'Compare Multimodal Outputs' : 'Compare'}
          </Button>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              Result
              {result.winner && winnerBadge(result.winner)}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {result.explanation && (
              <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                {result.explanation}
              </p>
            )}
            {result.confidence != null && (
              <div className="text-sm">
                <span className="text-muted-foreground">Confidence: </span>
                <span className="font-semibold">{result.confidence.toFixed(2)}</span>
              </div>
            )}
            <div className="flex gap-6 text-sm">
              {result.score_a != null && (
                <div>
                  <span className="text-muted-foreground">Score A: </span>
                  <span className="font-semibold">
                    {typeof result.score_a === 'number' ? result.score_a.toFixed(4) : String(result.score_a)}
                  </span>
                </div>
              )}
              {result.score_b != null && (
                <div>
                  <span className="text-muted-foreground">Score B: </span>
                  <span className="font-semibold">
                    {typeof result.score_b === 'number' ? result.score_b.toFixed(4) : String(result.score_b)}
                  </span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
