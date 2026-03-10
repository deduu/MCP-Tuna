import { useState } from 'react'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { DEFAULT_ORCHESTRATION_SETTINGS, useOrchestrationStore } from '@/stores/orchestration'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'

interface OrchestrationStepProps {
  stepIndex: number
  onComplete: (result: Record<string, unknown>) => void
}

const STEP_CONFIGS = [
  { title: 'Generate Problems', tool: 'orchestration.generate_problems' },
  { title: 'Collect Trajectories', tool: 'orchestration.collect_trajectories' },
  { title: 'Build Training Data', tool: 'orchestration.build_training_data' },
  { title: 'Train Orchestrator Model', tool: 'orchestration.train_orchestrator' },
] as const

const OUTPUT_FORMATS = ['sft', 'dpo', 'grpo'] as const

const PRESETS = [
  {
    id: 'starter',
    label: 'Starter',
    description: 'Recommended: simple SFT routing model with moderate data volume.',
    settings: {
      numProblems: 8,
      nPerProblem: 3,
      outputFormat: 'sft' as const,
      costBudget: 1,
      timeBudget: 60,
      numEpochs: 2,
      deploy: false,
      deployPort: DEFAULT_ORCHESTRATION_SETTINGS.deployPort,
    },
  },
  {
    id: 'preference',
    label: 'DPO',
    description: 'Use best-vs-worst trajectories to teach preferences between plans.',
    settings: {
      numProblems: 12,
      nPerProblem: 4,
      outputFormat: 'dpo' as const,
      costBudget: 1,
      timeBudget: 60,
      numEpochs: 2,
      deploy: false,
      deployPort: DEFAULT_ORCHESTRATION_SETTINGS.deployPort,
    },
  },
  {
    id: 'reward',
    label: 'GRPO',
    description: 'Use reward-weighted trajectories; higher variance and more exploratory.',
    settings: {
      numProblems: 8,
      nPerProblem: 4,
      outputFormat: 'grpo' as const,
      costBudget: 1,
      timeBudget: 90,
      numEpochs: 1,
      deploy: false,
      deployPort: DEFAULT_ORCHESTRATION_SETTINGS.deployPort,
    },
  },
] as const

function asList(input: unknown): Array<Record<string, unknown>> {
  if (Array.isArray(input)) return input as Array<Record<string, unknown>>
  if (input && typeof input === 'object' && Array.isArray((input as Record<string, unknown>).data)) {
    return (input as Record<string, unknown>).data as Array<Record<string, unknown>>
  }
  return []
}

export function OrchestrationStep({ stepIndex, onComplete }: OrchestrationStepProps) {
  const { stepResults, settings, setSettings } = useOrchestrationStore()
  const { mutate: execute, isPending } = useToolExecution()

  const config = STEP_CONFIGS[stepIndex]
  const [result, setResult] = useState<Record<string, unknown> | null>(null)

  function buildArgs(): Record<string, unknown> {
    switch (stepIndex) {
      case 0:
        return { domain_description: settings.domainDescription, num_problems: settings.numProblems }
      case 1: {
        const problems = asList(stepResults[0]?.problems ?? stepResults[0])
        return { problems, n_per_problem: settings.nPerProblem }
      }
      case 2: {
        const collected = asList(stepResults[1]?.collected ?? stepResults[1])
        return {
          collected,
          format: settings.outputFormat,
          cost_budget: settings.costBudget,
          time_budget: settings.timeBudget,
        }
      }
      case 3:
        return {
          domain_description: settings.domainDescription,
          output_dir: settings.outputDir,
          output_format: typeof stepResults[2]?.format === 'string' ? stepResults[2].format : settings.outputFormat,
          training_data: asList(stepResults[2]?.data ?? stepResults[2]),
          ...(settings.baseModel.trim() ? { base_model: settings.baseModel.trim() } : {}),
          num_epochs: settings.numEpochs,
          deploy: settings.deploy,
          deploy_port: settings.deployPort,
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
            ? ({
                success: true,
                data: payload,
                ...(stepIndex === 2 ? { format: settings.outputFormat } : {}),
              } as Record<string, unknown>)
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

      <div className="space-y-2 rounded-lg border border-border/70 bg-card/50 p-3">
        <p className="text-xs font-medium text-foreground">Recommended Presets</p>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              onClick={() => setSettings(preset.settings)}
              className={cn(
                'rounded-md border px-3 py-1.5 text-sm transition-colors',
                settings.outputFormat === preset.settings.outputFormat &&
                settings.numProblems === preset.settings.numProblems &&
                settings.nPerProblem === preset.settings.nPerProblem
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-input text-muted-foreground hover:text-foreground',
              )}
            >
              {preset.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          {PRESETS.find(
            (preset) =>
              settings.outputFormat === preset.settings.outputFormat &&
              settings.numProblems === preset.settings.numProblems &&
              settings.nPerProblem === preset.settings.nPerProblem,
          )?.description ?? 'Choose a preset to fill in sane defaults, then adjust as needed.'}
        </p>
      </div>

      {stepIndex === 0 && (
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label className="text-sm font-medium mb-1 block">Domain Description</label>
            <Input
              value={settings.domainDescription}
              onChange={(e) => setSettings({ domainDescription: e.target.value })}
              placeholder="customer support automation"
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Number of Problems</label>
            <Input
              type="number"
              value={settings.numProblems}
              onChange={(e) => setSettings({ numProblems: Number(e.target.value) })}
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
              value={settings.nPerProblem}
              onChange={(e) => setSettings({ nPerProblem: Number(e.target.value) })}
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
              value={settings.outputFormat}
              onChange={(e) => setSettings({ outputFormat: e.target.value as 'sft' | 'dpo' | 'grpo' })}
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
              value={settings.costBudget}
              onChange={(e) => setSettings({ costBudget: Number(e.target.value) })}
              step={0.1}
              min={0}
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Time Budget (s)</label>
            <Input
              type="number"
              value={settings.timeBudget}
              onChange={(e) => setSettings({ timeBudget: Number(e.target.value) })}
              min={1}
            />
          </div>
        </div>
      )}

      {stepIndex === 3 && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">
            This step trains from the dataset produced in Step 3. It does not regenerate problems or trajectories.
          </p>
          <div>
            <label className="text-sm font-medium mb-1 block">Output Directory</label>
            <Input
              value={settings.outputDir}
              onChange={(e) => setSettings({ outputDir: e.target.value })}
              placeholder="./output/orchestrator"
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Base Model (optional)</label>
            <Input
              value={settings.baseModel}
              onChange={(e) => setSettings({ baseModel: e.target.value })}
              placeholder="meta-llama/Llama-3.2-3B-Instruct"
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Epochs</label>
            <Input
              type="number"
              value={settings.numEpochs}
              onChange={(e) => setSettings({ numEpochs: Number(e.target.value) })}
              min={1}
            />
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={settings.deploy}
              onChange={(e) => setSettings({ deploy: e.target.checked })}
              className="h-4 w-4 rounded border-input bg-transparent"
            />
            Deploy after training
          </label>
          <div>
            <label className="text-sm font-medium mb-1 block">Deploy Port</label>
            <Input
              type="number"
              value={settings.deployPort}
              onChange={(e) => setSettings({ deployPort: Number(e.target.value) })}
              min={1}
              disabled={!settings.deploy}
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
