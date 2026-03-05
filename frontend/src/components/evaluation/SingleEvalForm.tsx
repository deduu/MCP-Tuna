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
  const [input, setInput] = useState('')
  const [output, setOutput] = useState('')
  const [rubric, setRubric] = useState('')
  const [result, setResult] = useState<EvalResult | null>(null)
  const { mutateAsync: executeTool, isPending } = useToolExecution()

  async function handleEvaluate() {
    if (!input.trim() || !output.trim()) {
      toast.error('Input and output are required')
      return
    }
    if (mode === 'rubric' && !rubric.trim()) {
      toast.error('Rubric is required for rubric evaluation')
      return
    }

    try {
      const toolName = mode === 'rubric' ? 'judge.evaluate_with_rubric' : 'judge.evaluate_response'
      const args: Record<string, unknown> = { input, output }
      if (mode === 'rubric') {
        args.rubric = rubric
      }
      const res = await executeTool({ toolName, args })
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
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-medium">Output</label>
            <textarea
              placeholder="The response to evaluate..."
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-24 resize-y"
              value={output}
              onChange={(e) => setOutput(e.target.value)}
            />
          </div>
          {mode === 'rubric' && (
            <div className="space-y-1">
              <label className="text-sm font-medium">Rubric</label>
              <textarea
                placeholder="Evaluation criteria / rubric text..."
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
              {result.score != null && (
                <span className="text-3xl font-bold text-primary">
                  {typeof result.score === 'number' ? result.score.toFixed(2) : String(result.score)}
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
