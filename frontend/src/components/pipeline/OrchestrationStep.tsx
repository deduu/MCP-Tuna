import { useState } from 'react'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { useOrchestrationStore } from '@/stores/orchestration'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { toast } from 'sonner'

interface OrchestrationStepProps {
  stepIndex: number
  onComplete: (result: Record<string, unknown>) => void
}

const STEP_CONFIGS = [
  { title: 'Generate Problems', tool: 'orchestration.generate_problems' },
  { title: 'Collect Trajectories', tool: 'orchestration.collect_trajectories' },
  { title: 'Build Training Data', tool: 'orchestration.build_training_data' },
  { title: 'Train Orchestrator', tool: 'orchestration.train_orchestrator' },
] as const

const OUTPUT_FORMATS = ['sft', 'dpo', 'grpo'] as const

function asList(input: unknown): Array<Record<string, unknown>> {
  if (Array.isArray(input)) return input as Array<Record<string, unknown>>
  if (input && typeof input === 'object' && Array.isArray((input as Record<string, unknown>).data)) {
    return (input as Record<string, unknown>).data as Array<Record<string, unknown>>
  }
  return []
}

export function OrchestrationStep({ stepIndex, onComplete }: OrchestrationStepProps) {
  const { stepResults } = useOrchestrationStore()
  const { mutate: execute, isPending } = useToolExecution()

  const config = STEP_CONFIGS[stepIndex]
  const [result, setResult] = useState<Record<string, unknown> | null>(null)

  const [domainDescription, setDomainDescription] = useState('general assistant workflows')
  const [numProblems, setNumProblems] = useState(10)
  const [nPerProblem, setNPerProblem] = useState(3)
  const [outputFormat, setOutputFormat] = useState<string>('sft')
  const [costBudget, setCostBudget] = useState(1)
  const [timeBudget, setTimeBudget] = useState(60)
  const [outputDir, setOutputDir] = useState('./output/orchestrator')
  const [baseModel, setBaseModel] = useState('')

  function buildArgs(): Record<string, unknown> {
    switch (stepIndex) {
      case 0:
        return { domain_description: domainDescription, num_problems: numProblems }
      case 1: {
        const problems = asList(stepResults[0]?.problems ?? stepResults[0])
        return { problems, n_per_problem: nPerProblem }
      }
      case 2: {
        const collected = asList(stepResults[1]?.collected ?? stepResults[1])
        return {
          collected,
          format: outputFormat,
          cost_budget: costBudget,
          time_budget: timeBudget,
        }
      }
      case 3:
        return {
          domain_description: domainDescription,
          num_problems: numProblems,
          n_per_problem: nPerProblem,
          output_dir: outputDir,
          output_format: outputFormat,
          ...(baseModel.trim() ? { base_model: baseModel.trim() } : {}),
        }
      default:
        return {}
    }
  }

  function handleExecute() {
    execute(
      { toolName: config.tool, args: buildArgs() },
      {
        onSuccess: (data) => {
          const payload = data as unknown
          const normalized = Array.isArray(payload)
            ? ({ success: true, data: payload } as Record<string, unknown>)
            : (payload as Record<string, unknown>)
          setResult(normalized)
          toast.success(`${config.title} completed`)
        },
        onError: (err) => toast.error(`${config.title} failed: ${err.message}`),
      },
    )
  }

  function handleNext() {
    if (result) onComplete(result)
  }

  return (
    <div className="space-y-4">
      <h3 className="text-base font-semibold">{config.title}</h3>

      {stepIndex === 0 && (
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label className="text-sm font-medium mb-1 block">Domain Description</label>
            <Input
              value={domainDescription}
              onChange={(e) => setDomainDescription(e.target.value)}
              placeholder="customer support automation"
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Number of Problems</label>
            <Input
              type="number"
              value={numProblems}
              onChange={(e) => setNumProblems(Number(e.target.value))}
              min={1}
            />
          </div>
        </div>
      )}

      {stepIndex === 1 && (
        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium mb-1 block">Trajectories per Problem</label>
            <Input
              type="number"
              value={nPerProblem}
              onChange={(e) => setNPerProblem(Number(e.target.value))}
              min={1}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Uses problem list from Step 1 output.
          </p>
        </div>
      )}

      {stepIndex === 2 && (
        <div className="grid gap-3 sm:grid-cols-3">
          <div>
            <label className="text-sm font-medium mb-1 block">Output Format</label>
            <select
              value={outputFormat}
              onChange={(e) => setOutputFormat(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {OUTPUT_FORMATS.map((f) => (
                <option key={f} value={f}>{f.toUpperCase()}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Cost Budget</label>
            <Input
              type="number"
              value={costBudget}
              onChange={(e) => setCostBudget(Number(e.target.value))}
              step={0.1}
              min={0}
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Time Budget (s)</label>
            <Input
              type="number"
              value={timeBudget}
              onChange={(e) => setTimeBudget(Number(e.target.value))}
              min={1}
            />
          </div>
        </div>
      )}

      {stepIndex === 3 && (
        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium mb-1 block">Output Directory</label>
            <Input
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              placeholder="./output/orchestrator"
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Base Model (optional)</label>
            <Input
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
              placeholder="meta-llama/Llama-3.2-3B-Instruct"
            />
          </div>
        </div>
      )}

      <div className="flex items-center gap-2">
        <Button onClick={handleExecute} disabled={isPending}>
          {isPending ? 'Executing...' : 'Execute Step'}
        </Button>
        {result && (
          <Button variant="secondary" onClick={handleNext}>
            Next
          </Button>
        )}
      </div>

      {result && (
        <Card>
          <CardContent className="p-3">
            <p className="text-xs font-medium text-muted-foreground mb-1">Result Preview</p>
            <pre className="max-h-40 overflow-auto text-xs font-mono">
              {JSON.stringify(result, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
