import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { toast } from 'sonner'

export function CriteriaManager() {
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const [rubricName, setRubricName] = useState('default-rubric')
  const [criterionName, setCriterionName] = useState('accuracy')
  const [description, setDescription] = useState('Factual correctness and completeness')
  const [weight, setWeight] = useState('1')
  const [minScore, setMinScore] = useState('1')
  const [maxScore, setMaxScore] = useState('10')
  const [lastResult, setLastResult] = useState<Record<string, unknown> | null>(null)

  async function handleCreateRubric() {
    if (!rubricName.trim() || !criterionName.trim() || !description.trim()) {
      toast.error('Rubric name, criterion name, and description are required')
      return
    }

    try {
      const criterion = {
        name: criterionName.trim(),
        description: description.trim(),
        weight: Number(weight || 1),
        min_score: Number(minScore || 1),
        max_score: Number(maxScore || 10),
      }
      const result = await executeTool({
        toolName: 'judge.create_rubric',
        args: {
          name: rubricName.trim(),
          description: '',
          criteria: [criterion],
        },
      })
      setLastResult(result as Record<string, unknown>)
      toast.success('Rubric validated')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to create rubric')
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="space-y-1">
          <label className="text-sm font-medium">Rubric Name</label>
          <Input value={rubricName} onChange={(e) => setRubricName(e.target.value)} />
        </div>
        <div className="space-y-1">
          <label className="text-sm font-medium">Criterion Name</label>
          <Input value={criterionName} onChange={(e) => setCriterionName(e.target.value)} />
        </div>
        <div className="space-y-1 sm:col-span-2">
          <label className="text-sm font-medium">Description</label>
          <Input value={description} onChange={(e) => setDescription(e.target.value)} />
        </div>
        <div className="space-y-1">
          <label className="text-sm font-medium">Weight</label>
          <Input type="number" value={weight} onChange={(e) => setWeight(e.target.value)} step="0.1" />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="space-y-1">
            <label className="text-sm font-medium">Min</label>
            <Input type="number" value={minScore} onChange={(e) => setMinScore(e.target.value)} />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-medium">Max</label>
            <Input type="number" value={maxScore} onChange={(e) => setMaxScore(e.target.value)} />
          </div>
        </div>
      </div>

      <Button size="sm" onClick={handleCreateRubric} disabled={isPending}>
        {isPending ? 'Validating...' : 'Validate Rubric'}
      </Button>

      {lastResult && (
        <pre className="max-h-56 overflow-auto rounded-md bg-secondary/40 p-3 text-xs font-mono">
          {JSON.stringify(lastResult, null, 2)}
        </pre>
      )}
    </div>
  )
}
