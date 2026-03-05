import { useState } from 'react'
import { Play, Settings2, ChevronDown } from 'lucide-react'
import { useRunFullPipeline, useRunPipeline } from '@/api/hooks/usePipeline'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { CustomPipelineForm } from './CustomPipelineForm'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'

const TECHNIQUES = ['sft', 'dpo', 'grpo', 'kto'] as const

export function PipelineTemplates() {
  // Full pipeline state
  const [docPath, setDocPath] = useState('')
  const [technique, setTechnique] = useState<string>('sft')
  const [modelPath, setModelPath] = useState('')
  const fullPipeline = useRunFullPipeline()

  // Custom pipeline
  const [customOpen, setCustomOpen] = useState(false)
  const customPipeline = useRunPipeline()

  function handleRunFull() {
    fullPipeline.mutate(
      { document_path: docPath, technique, model_path: modelPath },
      {
        onSuccess: () => toast.success('Full pipeline started'),
        onError: (err) => toast.error(`Pipeline failed: ${err.message}`),
      },
    )
  }

  function handleRunCustom(args: Record<string, unknown>) {
    customPipeline.mutate(args, {
      onSuccess: () => toast.success('Custom pipeline started'),
      onError: (err) => toast.error(`Pipeline failed: ${err.message}`),
    })
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {/* Full Pipeline */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-4 w-4" />
            Full Pipeline
          </CardTitle>
          <CardDescription>
            End-to-end: load &rarr; generate &rarr; clean &rarr; normalize &rarr; evaluate &rarr; train
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <label className="text-sm font-medium text-foreground mb-1 block">Document Path</label>
            <Input
              value={docPath}
              onChange={(e) => setDocPath(e.target.value)}
              placeholder="/path/to/documents"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-foreground mb-1 block">Technique</label>
            <select
              value={technique}
              onChange={(e) => setTechnique(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {TECHNIQUES.map((t) => (
                <option key={t} value={t}>
                  {t.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium text-foreground mb-1 block">Model Path</label>
            <Input
              value={modelPath}
              onChange={(e) => setModelPath(e.target.value)}
              placeholder="meta-llama/Llama-3-8B"
            />
          </div>
          <Button onClick={handleRunFull} disabled={fullPipeline.isPending}>
            {fullPipeline.isPending ? 'Starting...' : 'Run Full Pipeline'}
          </Button>
        </CardContent>
      </Card>

      {/* Custom Pipeline (collapsible) */}
      <Card>
        <CardHeader>
          <button
            type="button"
            onClick={() => setCustomOpen((v) => !v)}
            className="flex items-center gap-2 w-full text-left cursor-pointer"
          >
            <Settings2 className="h-4 w-4" />
            <CardTitle className="flex-1">Custom Pipeline</CardTitle>
            <ChevronDown
              className={cn('h-4 w-4 transition-transform text-muted-foreground', customOpen && 'rotate-180')}
            />
          </button>
          <CardDescription>
            Select individual steps and configure each one
          </CardDescription>
        </CardHeader>
        {customOpen && (
          <CardContent>
            <CustomPipelineForm
              onSubmit={handleRunCustom}
              isPending={customPipeline.isPending}
            />
          </CardContent>
        )}
      </Card>
    </div>
  )
}
