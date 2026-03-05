import { useState } from 'react'
import { Settings, Gauge } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useJudgeConfig } from '@/api/hooks/useEvaluation'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

export function JudgeConfig() {
  const { data: config, isLoading } = useJudgeConfig()
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const [editing, setEditing] = useState(false)
  const [formData, setFormData] = useState<Record<string, string>>({})
  const [calibrating, setCalibrating] = useState(false)

  function startEditing() {
    const initial: Record<string, string> = {}
    if (config) {
      for (const [k, v] of Object.entries(config)) {
        initial[k] = String(v ?? '')
      }
    }
    setFormData(initial)
    setEditing(true)
  }

  function updateField(key: string, value: string) {
    setFormData((prev) => ({ ...prev, [key]: value }))
  }

  async function handleSave() {
    try {
      await executeTool({ toolName: 'judge.configure_judge', args: formData })
      await queryClient.invalidateQueries({ queryKey: ['judge', 'config'] })
      toast.success('Judge configuration updated')
      setEditing(false)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to update config')
    }
  }

  async function handleCalibrate() {
    setCalibrating(true)
    try {
      await executeTool({ toolName: 'judge.calibrate', args: {} })
      toast.success('Judge calibration complete')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Calibration failed')
    } finally {
      setCalibrating(false)
    }
  }

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading configuration...</p>
  }

  if (editing) {
    return (
      <div className="space-y-4">
        {Object.entries(formData).map(([key, value]) => (
          <div key={key} className="space-y-1">
            <label className="text-sm font-medium">{key}</label>
            <Input
              value={value}
              onChange={(e) => updateField(key, e.target.value)}
            />
          </div>
        ))}
        <div className="flex gap-2">
          <Button onClick={handleSave} disabled={isPending}>
            {isPending ? 'Saving...' : 'Save'}
          </Button>
          <Button variant="ghost" onClick={() => setEditing(false)}>
            Cancel
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {config && Object.keys(config).length > 0 ? (
        <div className="grid grid-cols-2 gap-2 text-sm">
          {Object.entries(config).map(([key, value]) => (
            <div key={key} className="flex gap-2">
              <span className="text-muted-foreground">{key}:</span>
              <span className="font-medium">{String(value ?? '-')}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">No configuration loaded</p>
      )}
      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={startEditing}>
          <Settings className="h-3.5 w-3.5" />
          Configure
        </Button>
        <Button variant="outline" size="sm" onClick={handleCalibrate} disabled={calibrating}>
          <Gauge className="h-3.5 w-3.5" />
          {calibrating ? 'Calibrating...' : 'Calibrate'}
        </Button>
      </div>
    </div>
  )
}
