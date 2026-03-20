import { useEffect, useMemo, useState } from 'react'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

type StepConfigValue = Record<string, Record<string, unknown>>
type EditorMode = 'form' | 'json'

interface StepFieldConfig {
  key: string
  label: string
  type: 'integer' | 'number' | 'boolean'
  defaultValue: number | boolean
  description?: string
}

interface StepConfigEditorProps {
  selectedSteps: string[]
  value: StepConfigValue
  onChange: (value: StepConfigValue, isValid: boolean) => void
}

const STEP_FIELD_CONFIG: Partial<Record<string, StepFieldConfig[]>> = {
  train: [
    { key: 'num_epochs', label: 'num_epochs', type: 'integer', defaultValue: 3 },
    { key: 'learning_rate', label: 'learning_rate', type: 'number', defaultValue: 2e-4 },
    { key: 'per_device_train_batch_size', label: 'per_device_train_batch_size', type: 'integer', defaultValue: 1 },
    { key: 'gradient_accumulation_steps', label: 'gradient_accumulation_steps', type: 'integer', defaultValue: 4 },
    { key: 'max_seq_length', label: 'max_seq_length', type: 'integer', defaultValue: 2048 },
    { key: 'warmup_ratio', label: 'warmup_ratio', type: 'number', defaultValue: 0 },
    { key: 'weight_decay', label: 'weight_decay', type: 'number', defaultValue: 0 },
    { key: 'lora_r', label: 'lora_r', type: 'integer', defaultValue: 8 },
    { key: 'lora_alpha', label: 'lora_alpha', type: 'integer', defaultValue: 16 },
    {
      key: 'completion_only_loss',
      label: 'completion_only_loss',
      type: 'boolean',
      defaultValue: true,
      description: 'Train only on assistant tokens when supported by the installed TRL stack.',
    },
    {
      key: 'enable_evaluation',
      label: 'enable_evaluation',
      type: 'boolean',
      defaultValue: true,
    },
    {
      key: 'save_best_model',
      label: 'save_best_model',
      type: 'boolean',
      defaultValue: true,
    },
    { key: 'load_in_4bit', label: 'load_in_4bit', type: 'boolean', defaultValue: true },
  ],
  deploy: [
    {
      key: 'deploy_port',
      label: 'deploy_port',
      type: 'integer',
      defaultValue: 8001,
      description: 'Port to use when deployment runs after training.',
    },
  ],
}

function normalizeStepConfig(value: StepConfigValue): StepConfigValue {
  return JSON.parse(JSON.stringify(value || {})) as StepConfigValue
}

function formatJson(value: StepConfigValue): string {
  if (!value || Object.keys(value).length === 0) return ''
  return JSON.stringify(value, null, 2)
}

function parseNumber(raw: string, type: StepFieldConfig['type']): number | undefined {
  if (!raw.trim()) return undefined
  const parsed = type === 'integer' ? parseInt(raw, 10) : parseFloat(raw)
  return Number.isFinite(parsed) ? parsed : undefined
}

function StepField({
  field,
  value,
  onChange,
}: {
  field: StepFieldConfig
  value: unknown
  onChange: (nextValue: unknown) => void
}) {
  if (field.type === 'boolean') {
    return (
      <label className="flex items-start gap-2 rounded-md border border-input/60 p-3 text-sm">
        <input
          type="checkbox"
          checked={value === true}
          onChange={(event) => onChange(event.target.checked ? true : undefined)}
          className="mt-0.5 rounded border-input"
        />
        <span className="space-y-1">
          <span className="block font-medium">{field.label}</span>
          <span className="block text-xs text-muted-foreground">
            Default: {String(field.defaultValue)}
            {field.description ? ` · ${field.description}` : ''}
          </span>
        </span>
      </label>
    )
  }

  return (
    <div className="space-y-1">
      <label className="text-xs text-muted-foreground">{field.label}</label>
      <Input
        type="number"
        step={field.type === 'integer' ? '1' : 'any'}
        value={value === undefined ? '' : String(value)}
        onChange={(event) => onChange(parseNumber(event.target.value, field.type))}
        placeholder={String(field.defaultValue)}
      />
      <p className="text-xs text-muted-foreground">
        Default: {String(field.defaultValue)}
        {field.description ? ` · ${field.description}` : ''}
      </p>
    </div>
  )
}

export function StepConfigEditor({ selectedSteps, value, onChange }: StepConfigEditorProps) {
  const [mode, setMode] = useState<EditorMode>('form')
  const [rawValue, setRawValue] = useState(formatJson(value))
  const [jsonError, setJsonError] = useState<string | null>(null)

  const visibleSteps = useMemo(
    () => selectedSteps.filter((step) => (STEP_FIELD_CONFIG[step] ?? []).length > 0),
    [selectedSteps],
  )

  useEffect(() => {
    setRawValue(formatJson(value))
  }, [value])

  const updateConfig = (mutator: (draft: StepConfigValue) => void) => {
    const next = normalizeStepConfig(value)
    mutator(next)
    onChange(next, true)
  }

  const updateField = (step: string, key: string, nextValue: unknown) => {
    updateConfig((draft) => {
      const current = { ...(draft[step] ?? {}) }
      if (nextValue === undefined || nextValue === '') {
        delete current[key]
      } else {
        current[key] = nextValue
      }
      if (Object.keys(current).length === 0) {
        delete draft[step]
      } else {
        draft[step] = current
      }
    })
  }

  const applyStepDefaults = (step: string) => {
    updateConfig((draft) => {
      const next: Record<string, unknown> = { ...(draft[step] ?? {}) }
      for (const field of STEP_FIELD_CONFIG[step] ?? []) {
        next[field.key] = field.defaultValue
      }
      draft[step] = next
    })
  }

  const handleRawChange = (nextRaw: string) => {
    setRawValue(nextRaw)
    if (!nextRaw.trim()) {
      setJsonError(null)
      onChange({}, true)
      return
    }
    try {
      const parsed = JSON.parse(nextRaw)
      if (parsed === null || Array.isArray(parsed) || typeof parsed !== 'object') {
        setJsonError('step_config must be a JSON object')
        onChange(value, false)
        return
      }
      setJsonError(null)
      onChange(parsed as StepConfigValue, true)
    } catch {
      setJsonError('Invalid JSON')
      onChange(value, false)
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="space-y-1">
          <label className="text-sm font-medium">step_config</label>
          <p className="text-xs text-muted-foreground">
            Use the form for common overrides. Switch to JSON for exact payload control or unsupported parameters.
          </p>
        </div>
        <div className="flex items-center gap-1 rounded-md border border-input p-1">
          <button
            type="button"
            onClick={() => setMode('form')}
            className={cn(
              'rounded px-2 py-1 text-xs transition-colors',
              mode === 'form' ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground',
            )}
          >
            Form
          </button>
          <button
            type="button"
            onClick={() => setMode('json')}
            className={cn(
              'rounded px-2 py-1 text-xs transition-colors',
              mode === 'json' ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground',
            )}
          >
            JSON
          </button>
        </div>
      </div>

      {mode === 'form' ? (
        visibleSteps.length > 0 ? (
          <div className="space-y-4">
            {visibleSteps.map((step) => (
              <div key={step} className="space-y-3 rounded-md border border-input/60 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium capitalize">{step}</p>
                    <p className="text-xs text-muted-foreground">
                      Known overrides for the `{step}` step. Unknown keys are preserved if you add them in JSON mode.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => applyStepDefaults(step)}
                    className="rounded border border-input px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
                  >
                    Apply defaults
                  </button>
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  {(STEP_FIELD_CONFIG[step] ?? []).map((field) => (
                    <StepField
                      key={`${step}.${field.key}`}
                      field={field}
                      value={value[step]?.[field.key]}
                      onChange={(nextValue) => updateField(step, field.key, nextValue)}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-md border border-dashed border-input p-3 text-sm text-muted-foreground">
            No schema-aware overrides are defined for the currently selected steps. Use JSON mode if you need a raw override object.
          </div>
        )
      ) : (
        <div className="space-y-2">
          <textarea
            value={rawValue}
            onChange={(event) => handleRawChange(event.target.value)}
            placeholder='{"train":{"num_epochs":1,"max_seq_length":512},"deploy":{"deploy_port":8001}}'
            rows={8}
            className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono resize-y"
          />
          <div className="flex items-center justify-between gap-3">
            <p className="text-xs text-muted-foreground">
              Raw JSON accepts either step ids like `train` or full tool names like `generate.from_document`.
            </p>
            {jsonError && <p className="text-xs text-red-400">{jsonError}</p>}
          </div>
        </div>
      )}
    </div>
  )
}
