import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import type { ChatImageBlock } from '@/lib/chat-content'
import { buildEvaluationMessages } from '@/lib/evaluation-multimodal'
import { toast } from 'sonner'
import { ImageAttachmentField } from './ImageAttachmentField'

interface SingleEvalFormProps {
  mode: 'single' | 'rubric'
}

interface EvalResult {
  overallScore?: number
  feedback?: string
  breakdown?: Record<string, number | string>
  raw?: Record<string, unknown>
  [key: string]: unknown
}

function normalizeEvalResult(payload: Record<string, unknown>): EvalResult {
  const result = payload.result && typeof payload.result === 'object'
    ? (payload.result as Record<string, unknown>)
    : payload
  const criteriaScores = Array.isArray(result.criteria_scores)
    ? (result.criteria_scores as Array<Record<string, unknown>>)
    : []

  const breakdown: Record<string, number | string> = {}
  for (const item of criteriaScores) {
    const criterion = typeof item.criterion === 'string' ? item.criterion : null
    if (!criterion) continue
    breakdown[criterion] = typeof item.score === 'number' ? item.score : String(item.score ?? '')
  }

  return {
    overallScore: typeof result.overall_score === 'number' ? result.overall_score : undefined,
    feedback: typeof result.error === 'string' && result.error.trim()
      ? result.error
      : criteriaScores
          .map((item) => {
            const criterion = typeof item.criterion === 'string' ? item.criterion : ''
            const reason = typeof item.reason === 'string' ? item.reason.trim() : ''
            return criterion && reason ? `${criterion}: ${reason}` : ''
          })
          .filter(Boolean)
          .join('\n'),
    breakdown,
    raw: result,
  }
}

export function SingleEvalForm({ mode }: SingleEvalFormProps) {
  const [question, setQuestion] = useState('')
  const [generated, setGenerated] = useState('')
  const [reference, setReference] = useState('')
  const [rubric, setRubric] = useState('')
  const [images, setImages] = useState<ChatImageBlock[]>([])
  const [isUploadingImages, setIsUploadingImages] = useState(false)
  const [result, setResult] = useState<EvalResult | null>(null)
  const { mutateAsync: executeTool, isPending } = useToolExecution()

  async function handleEvaluate() {
    if ((!question.trim() && images.length === 0) || !generated.trim()) {
      toast.error('A prompt or image plus an output is required')
      return
    }

    try {
      const args: Record<string, unknown> = {
        question: question.trim(),
        generated: generated.trim(),
        judge_type: mode === 'rubric' ? 'rubric' : 'pointwise',
      }
      if (reference.trim()) {
        args.reference = reference.trim()
      }
      if (mode === 'rubric') {
        if (!rubric.trim()) {
          toast.error('Rubric JSON is required for rubric evaluation')
          return
        }
        try {
          args.rubric = JSON.parse(rubric)
        } catch {
          toast.error('Rubric must be valid JSON')
          return
        }
      }
      const toolName = images.length > 0 ? 'judge.evaluate_vlm' : 'judge.evaluate'
      const res = await executeTool({
        toolName,
        args: images.length > 0
          ? {
              messages: buildEvaluationMessages(question, images),
              generated: generated.trim(),
              ...(reference.trim() ? { reference: reference.trim() } : {}),
              ...(mode === 'rubric' ? { rubric: args.rubric } : {}),
            }
          : args,
      })
      setResult(normalizeEvalResult(res as Record<string, unknown>))
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Evaluation failed')
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="pt-6 space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">Input</label>
            <textarea
              placeholder={images.length > 0 ? 'Prompt text to pair with the attached images...' : 'The input/prompt to evaluate against...'}
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-24 resize-y"
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
            <label className="text-sm font-medium">Output</label>
            <textarea
              placeholder="The response to evaluate..."
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-24 resize-y"
              value={generated}
              onChange={(e) => setGenerated(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-medium">Reference (optional)</label>
            <textarea
              placeholder="Reference answer..."
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-20 resize-y"
              value={reference}
              onChange={(e) => setReference(e.target.value)}
            />
          </div>
          {mode === 'rubric' && (
            <div className="space-y-1">
              <label className="text-sm font-medium">Rubric</label>
              <textarea
                placeholder='{"name":"custom","criteria":[{"name":"accuracy","description":"Factual correctness","min_score":1,"max_score":10,"weight":1.0}]}'
                className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-20 resize-y"
                value={rubric}
                onChange={(e) => setRubric(e.target.value)}
              />
            </div>
          )}
          <Button onClick={handleEvaluate} disabled={isPending || isUploadingImages}>
            {isPending ? 'Evaluating...' : images.length > 0 ? 'Evaluate Multimodal Sample' : 'Evaluate'}
          </Button>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              Score
              {result.overallScore != null && (
                <span className="text-3xl font-bold text-primary">
                  {result.overallScore.toFixed(2)}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {result.feedback && (
              <div className="space-y-1">
                <p className="text-sm font-medium">Feedback</p>
                <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                  {result.feedback}
                </p>
              </div>
            )}
            {result.breakdown && Object.keys(result.breakdown).length > 0 && (
              <div className="space-y-1">
                <p className="text-sm font-medium">Detailed Breakdown</p>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  {Object.entries(result.breakdown).map(([key, value]) => (
                    <div key={key} className="flex gap-2">
                      <span className="text-muted-foreground">{key}:</span>
                      <span className="font-medium">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
