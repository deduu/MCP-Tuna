import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

const PIPELINE_STEPS = [
  'extract',
  'generate',
  'clean',
  'normalize',
  'evaluate',
  'train',
  'deploy',
] as const

const TECHNIQUES = ['sft', 'dpo', 'grpo', 'kto'] as const

interface CustomPipelineFormProps {
  onSubmit: (args: Record<string, unknown>) => void
  isPending: boolean
}

export function CustomPipelineForm({ onSubmit, isPending }: CustomPipelineFormProps) {
  const [selectedSteps, setSelectedSteps] = useState<Set<string>>(new Set())
  const [documentPath, setDocumentPath] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [technique, setTechnique] = useState<string>('sft')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [stepConfig, setStepConfig] = useState('')

  function toggleStep(step: string) {
    setSelectedSteps((prev) => {
      const next = new Set(prev)
      if (next.has(step)) next.delete(step)
      else next.add(step)
      return next
    })
  }

  function handleSubmit() {
    const args: Record<string, unknown> = {
      steps: Array.from(selectedSteps),
      document_path: documentPath,
      model_path: modelPath,
      technique,
    }
    if (stepConfig.trim()) {
      try {
        args.step_config = JSON.parse(stepConfig)
      } catch {
        // keep as string if invalid JSON
        args.step_config = stepConfig
      }
    }
    onSubmit(args)
  }

  return (
    <div className="space-y-4">
      {/* Step selection */}
      <div>
        <label className="text-sm font-medium text-foreground mb-2 block">Steps</label>
        <div className="flex flex-wrap gap-2">
          {PIPELINE_STEPS.map((step) => (
            <label
              key={step}
              className={cn(
                'flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm cursor-pointer transition-colors',
                selectedSteps.has(step)
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-input text-muted-foreground hover:text-foreground',
              )}
            >
              <input
                type="checkbox"
                checked={selectedSteps.has(step)}
                onChange={() => toggleStep(step)}
                className="sr-only"
              />
              {step}
            </label>
          ))}
        </div>
      </div>

      {/* Inputs */}
      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <label className="text-sm font-medium text-foreground mb-1 block">Document Path</label>
          <Input
            value={documentPath}
            onChange={(e) => setDocumentPath(e.target.value)}
            placeholder="/path/to/documents"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-foreground mb-1 block">Model Path</label>
          <Input
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            placeholder="meta-llama/Llama-3-8B"
          />
        </div>
      </div>

      <div>
        <label className="text-sm font-medium text-foreground mb-1 block">Technique</label>
        <select
          value={technique}
          onChange={(e) => setTechnique(e.target.value)}
          className="flex h-9 w-full max-w-xs rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          {TECHNIQUES.map((t) => (
            <option key={t} value={t}>
              {t.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      {/* Advanced */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground cursor-pointer"
        >
          <ChevronDown
            className={cn('h-3.5 w-3.5 transition-transform', showAdvanced && 'rotate-180')}
          />
          Advanced: step_config
        </button>
        {showAdvanced && (
          <textarea
            value={stepConfig}
            onChange={(e) => setStepConfig(e.target.value)}
            placeholder='{"generate": {"temperature": 0.8}}'
            rows={4}
            className="mt-2 w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
        )}
      </div>

      <Button
        onClick={handleSubmit}
        disabled={isPending || selectedSteps.size === 0}
      >
        {isPending ? 'Running...' : 'Run Pipeline'}
      </Button>
    </div>
  )
}
