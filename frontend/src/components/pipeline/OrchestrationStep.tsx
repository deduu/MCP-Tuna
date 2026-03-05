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
  {
    title: 'Generate Problems',
    tool: 'orchestration.generate_problems',
    fields: ['domain', 'num_problems', 'difficulty'],
  },
  {
    title: 'Collect Trajectories',
    tool: 'orchestration.collect_trajectories',
    fields: ['problem_set_path', 'agent_config'],
  },
  {
    title: 'Score Trajectories',
    tool: 'orchestration.score_trajectories',
    fields: ['trajectory_path', 'weights'],
  },
  {
    title: 'Format Dataset',
    tool: 'orchestration.format_dataset',
    fields: ['scored_path', 'output_format'],
  },
] as const

const DIFFICULTIES = ['easy', 'medium', 'hard'] as const
const OUTPUT_FORMATS = ['sft', 'dpo', 'grpo'] as const

export function OrchestrationStep({ stepIndex, onComplete }: OrchestrationStepProps) {
  const { stepResults } = useOrchestrationStore()
  const { mutate: execute, isPending } = useToolExecution()

  const config = STEP_CONFIGS[stepIndex]
  const [result, setResult] = useState<Record<string, unknown> | null>(null)

  // Step 0 - Generate Problems
  const [domain, setDomain] = useState('')
  const [numProblems, setNumProblems] = useState(10)
  const [difficulty, setDifficulty] = useState<string>('medium')

  // Step 1 - Collect Trajectories
  const prevProblemPath = (stepResults[0]?.problem_set_path as string) ?? ''
  const [problemSetPath, setProblemSetPath] = useState(prevProblemPath)
  const [agentConfig, setAgentConfig] = useState('')

  // Step 2 - Score Trajectories
  const prevTrajectoryPath = (stepResults[1]?.trajectory_path as string) ?? ''
  const [trajectoryPath, setTrajectoryPath] = useState(prevTrajectoryPath)
  const [accuracy, setAccuracy] = useState(0.5)
  const [cost, setCost] = useState(0.3)
  const [latency, setLatency] = useState(0.2)

  // Step 3 - Format Dataset
  const prevScoredPath = (stepResults[2]?.scored_path as string) ?? ''
  const [scoredPath, setScoredPath] = useState(prevScoredPath)
  const [outputFormat, setOutputFormat] = useState<string>('sft')

  function buildArgs(): Record<string, unknown> {
    switch (stepIndex) {
      case 0:
        return { domain, num_problems: numProblems, difficulty }
      case 1: {
        const args: Record<string, unknown> = { problem_set_path: problemSetPath }
        if (agentConfig.trim()) {
          try {
            args.agent_config = JSON.parse(agentConfig)
          } catch {
            args.agent_config = agentConfig
          }
        }
        return args
      }
      case 2:
        return {
          trajectory_path: trajectoryPath,
          weights: { accuracy, cost, latency },
        }
      case 3:
        return { scored_path: scoredPath, output_format: outputFormat }
      default:
        return {}
    }
  }

  function handleExecute() {
    execute(
      { toolName: config.tool, args: buildArgs() },
      {
        onSuccess: (data) => {
          const res = data as unknown as Record<string, unknown>
          setResult(res)
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

      {/* Step 0: Generate Problems */}
      {stepIndex === 0 && (
        <div className="grid gap-3 sm:grid-cols-3">
          <div>
            <label className="text-sm font-medium mb-1 block">Domain</label>
            <Input value={domain} onChange={(e) => setDomain(e.target.value)} placeholder="mathematics" />
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
          <div>
            <label className="text-sm font-medium mb-1 block">Difficulty</label>
            <select
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {DIFFICULTIES.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Step 1: Collect Trajectories */}
      {stepIndex === 1 && (
        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium mb-1 block">Problem Set Path</label>
            <Input
              value={problemSetPath}
              onChange={(e) => setProblemSetPath(e.target.value)}
              placeholder="/path/to/problem_set"
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Agent Config (JSON)</label>
            <textarea
              value={agentConfig}
              onChange={(e) => setAgentConfig(e.target.value)}
              placeholder='{"model": "gpt-4", "temperature": 0.7}'
              rows={3}
              className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
          </div>
        </div>
      )}

      {/* Step 2: Score Trajectories */}
      {stepIndex === 2 && (
        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium mb-1 block">Trajectory Path</label>
            <Input
              value={trajectoryPath}
              onChange={(e) => setTrajectoryPath(e.target.value)}
              placeholder="/path/to/trajectories"
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1 block">Weights</label>
            <div className="grid gap-3 sm:grid-cols-3">
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Accuracy</label>
                <Input
                  type="number"
                  step={0.1}
                  min={0}
                  max={1}
                  value={accuracy}
                  onChange={(e) => setAccuracy(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Cost</label>
                <Input
                  type="number"
                  step={0.1}
                  min={0}
                  max={1}
                  value={cost}
                  onChange={(e) => setCost(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Latency</label>
                <Input
                  type="number"
                  step={0.1}
                  min={0}
                  max={1}
                  value={latency}
                  onChange={(e) => setLatency(Number(e.target.value))}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Step 3: Format Dataset */}
      {stepIndex === 3 && (
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label className="text-sm font-medium mb-1 block">Scored Path</label>
            <Input
              value={scoredPath}
              onChange={(e) => setScoredPath(e.target.value)}
              placeholder="/path/to/scored"
            />
          </div>
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
        </div>
      )}

      {/* Execute */}
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

      {/* Result preview */}
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
