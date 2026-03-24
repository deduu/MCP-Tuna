import { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { FlaskConical, GitCompare } from 'lucide-react'

interface InferenceTestProps {
  modelPath: string
  adapterPath?: string
}

function resolveTemperature(value: string): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return 0
  return Math.max(0, Math.min(2, parsed))
}

function resolveTopP(value: string): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return 0.9
  return Math.max(0, Math.min(1, parsed))
}

function resolveTopK(value: string): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return 50
  return Math.max(1, Math.round(parsed))
}

export function InferenceTest({ modelPath, adapterPath }: InferenceTestProps) {
  const [prompt, setPrompt] = useState('')
  const [baseModelPath, setBaseModelPath] = useState(adapterPath ? modelPath : '')
  const [temperature, setTemperature] = useState('0')
  const [topP, setTopP] = useState('0.9')
  const [topK, setTopK] = useState('50')
  const evaluateMutation = useToolExecution()
  const compareMutation = useToolExecution()

  useEffect(() => {
    setBaseModelPath(adapterPath ? modelPath : '')
  }, [adapterPath, modelPath])

  const handleEvaluate = () => {
    if (!prompt.trim()) return
    evaluateMutation.mutate({
      toolName: 'test.inference',
      args: {
        model_path: modelPath,
        ...(adapterPath ? { adapter_path: adapterPath } : {}),
        prompts: [prompt.trim()],
        temperature: resolveTemperature(temperature),
        top_p: resolveTopP(topP),
        top_k: resolveTopK(topK),
      },
    })
  }

  const handleCompare = () => {
    const resolvedBaseModelPath = (adapterPath ? modelPath : baseModelPath).trim()
    const resolvedAdapterPath = adapterPath?.trim() || modelPath.trim()
    if (!resolvedBaseModelPath || !resolvedAdapterPath) return
    compareMutation.mutate({
      toolName: 'test.compare_models',
      args: {
        prompts: [prompt.trim() || 'Hello'],
        base_model_path: resolvedBaseModelPath,
        finetuned_adapter_path: resolvedAdapterPath,
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
            placeholder={adapterPath ? 'Base model path (auto-filled from training job)' : 'Base model path (for compare)'}
            value={baseModelPath}
            onChange={(e) => setBaseModelPath(e.target.value)}
            disabled={Boolean(adapterPath)}
          />
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Temperature</label>
              <Input
                type="number"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                placeholder="0.0"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Top P</label>
              <Input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={topP}
                onChange={(e) => setTopP(e.target.value)}
                placeholder="0.9"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Top K</label>
              <Input
                type="number"
                min="1"
                step="1"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                placeholder="50"
              />
            </div>
          </div>
          {adapterPath && (
            <p className="text-xs text-muted-foreground">
              LoRA job detected. Inference uses the base model plus adapter automatically.
            </p>
          )}
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
              disabled={isLoading || (!adapterPath && !baseModelPath.trim())}
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
