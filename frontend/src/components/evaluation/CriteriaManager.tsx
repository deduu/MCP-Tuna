import { useState } from 'react'
import { Plus } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useJudgeCriteria } from '@/api/hooks/useEvaluation'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

export function CriteriaManager() {
  const { data: criteria, isLoading } = useJudgeCriteria()
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const [showAdd, setShowAdd] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [weight, setWeight] = useState('')

  async function handleCreate() {
    if (!name.trim() || !description.trim()) {
      toast.error('Name and description are required')
      return
    }
    try {
      const args: Record<string, unknown> = { name, description }
      if (weight.trim()) {
        args.weight = parseFloat(weight)
      }
      await executeTool({ toolName: 'judge.create_criterion', args })
      await queryClient.invalidateQueries({ queryKey: ['judge', 'criteria'] })
      toast.success(`Criterion "${name}" created`)
      setName('')
      setDescription('')
      setWeight('')
      setShowAdd(false)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to create criterion')
    }
  }

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading criteria...</p>
  }

  return (
    <div className="space-y-4">
      {criteria && criteria.length > 0 ? (
        <div className="flex flex-wrap gap-3">
          {criteria.map((c) => (
            <div key={c.name} className="flex items-center gap-2">
              <Badge variant="secondary">{c.name}</Badge>
              <span className="text-xs text-muted-foreground max-w-48 truncate">
                {c.description}
              </span>
              {c.weight != null && (
                <span className="text-xs text-muted-foreground">
                  (w: {c.weight})
                </span>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">No criteria defined</p>
      )}

      {showAdd ? (
        <div className="space-y-3 rounded-lg border border-border p-4">
          <div className="space-y-1">
            <label className="text-sm font-medium">Name</label>
            <Input
              placeholder="e.g. accuracy"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-medium">Description</label>
            <textarea
              placeholder="Describe this criterion..."
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring min-h-20 resize-y"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-medium">Weight (optional)</label>
            <Input
              type="number"
              step="0.1"
              placeholder="1.0"
              value={weight}
              onChange={(e) => setWeight(e.target.value)}
              className="max-w-32"
            />
          </div>
          <div className="flex gap-2">
            <Button size="sm" onClick={handleCreate} disabled={isPending}>
              {isPending ? 'Creating...' : 'Create'}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setShowAdd(false)}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <Button variant="outline" size="sm" onClick={() => setShowAdd(true)}>
          <Plus className="h-3.5 w-3.5" />
          Add Criterion
        </Button>
      )}
    </div>
  )
}
