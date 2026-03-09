import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { toast } from 'sonner'

interface SingleEvalFormProps {
  mode: 'single' | 'rubric'
}

interface EvalResult {
  score?: number
  feedback?: string
  breakdown?: Record<string, unknown>
  [key: string]: unknown
}

export function SingleEvalForm({ mode }: SingleEvalFormProps) {
  const [question, setQuestion] = useState('')
  const [generated, setGenerated] = useState('')
  const [reference, setReference] = useState('')
  const [rubric, setRubric] = useState('')
  const [result, setResult] = useState<EvalResult | null>(null)
  const { mutateAsync: executeTool, isPending } = useToolExecution()

  async function handleEvaluate() {
    if (!question.trim() || !generated.trim()) {
      toast.error('Input and output are required')
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
      const res = await executeTool({ toolName: 'judge.evaluate', args })
      setResult(res as EvalResult)
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
              placeholder="The input/prompt to evaluate against..."
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-24 resize-y"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>
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
          <Button onClick={handleEvaluate} disabled={isPending}>
            {isPending ? 'Evaluating...' : 'Evaluate'}
          </Button>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              Score
              {(result.score != null || (result as Record<string, unknown>).overall_score != null) && (
                <span className="text-3xl font-bold text-primary">
                  {(() => {
                    const raw = result.score ?? (result as Record<string, unknown>).overall_score
                    return typeof raw === 'number' ? raw.toFixed(2) : String(raw)
                  })()}
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
