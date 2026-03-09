import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { FlaskConical, GitCompare } from 'lucide-react'

interface InferenceTestProps {
  modelPath: string
}

export function InferenceTest({ modelPath }: InferenceTestProps) {
  const [prompt, setPrompt] = useState('')
  const [baseModelPath, setBaseModelPath] = useState('')
  const evaluateMutation = useToolExecution()
  const compareMutation = useToolExecution()

  const handleEvaluate = () => {
    if (!prompt.trim()) return
    evaluateMutation.mutate({
      toolName: 'test.inference',
      args: { model_path: modelPath, prompts: [prompt.trim()] },
    })
  }

  const handleCompare = () => {
    if (!baseModelPath.trim()) return
    compareMutation.mutate({
      toolName: 'test.compare_models',
      args: {
        prompts: [prompt.trim() || 'Hello'],
        base_model_path: baseModelPath.trim(),
        finetuned_adapter_path: modelPath,
      },
    })
  }

  const result = evaluateMutation.data ?? compareMutation.data
  const isLoading = evaluateMutation.isPending || compareMutation.isPending

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Test Inference</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-3">
          <textarea
            className="flex min-h-[100px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring resize-none"
            placeholder="Enter a prompt to test the deployed model..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <input
            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            placeholder="Base model path (for compare)"
            value={baseModelPath}
            onChange={(e) => setBaseModelPath(e.target.value)}
          />
          <div className="flex items-center gap-2">
            <Button
              onClick={handleEvaluate}
              disabled={isLoading || !prompt.trim()}
              size="sm"
            >
              <FlaskConical className="h-4 w-4" />
              {evaluateMutation.isPending ? 'Evaluating...' : 'Evaluate'}
            </Button>
            <Button
              variant="outline"
              onClick={handleCompare}
              disabled={isLoading || !baseModelPath.trim()}
              size="sm"
            >
              <GitCompare className="h-4 w-4" />
              {compareMutation.isPending ? 'Comparing...' : 'Compare'}
            </Button>
          </div>

          {result && (
            <Card className="bg-secondary/30">
              <CardContent className="p-4">
                <pre className="text-xs font-mono whitespace-pre-wrap break-all">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </CardContent>
            </Card>
          )}

          {(evaluateMutation.error || compareMutation.error) && (
            <p className="text-sm text-destructive">
              {(evaluateMutation.error ?? compareMutation.error)?.message}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
