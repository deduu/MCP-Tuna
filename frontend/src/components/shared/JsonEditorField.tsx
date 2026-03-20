import { useEffect, useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

type JsonScalar = string | number | boolean | null
export type JsonEditorValue =
  | JsonScalar
  | JsonEditorValue[]
  | { [key: string]: JsonEditorValue }

type JsonEditorMode = 'structured' | 'json'
type JsonValueType = 'object' | 'array' | 'string' | 'number' | 'boolean' | 'null'

export interface JsonEditorChange {
  parsed: JsonEditorValue | null
  raw: string
  isValid: boolean
  error: string | null
}

interface JsonEditorFieldProps {
  label: string
  description?: string
  placeholder?: string
  initialValue?: JsonEditorValue | null
  defaultValue?: JsonEditorValue
  allowEmpty?: boolean
  onChange?: (change: JsonEditorChange) => void
  className?: string
}

function getJsonValueType(value: JsonEditorValue): JsonValueType {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  switch (typeof value) {
    case 'string':
      return 'string'
    case 'number':
      return 'number'
    case 'boolean':
      return 'boolean'
    default:
      return 'object'
  }
}

function createDefaultValue(type: JsonValueType): JsonEditorValue {
  switch (type) {
    case 'array':
      return []
    case 'string':
      return ''
    case 'number':
      return 0
    case 'boolean':
      return false
    case 'null':
      return null
    default:
      return {}
  }
}

function normalizeInitialValue(
  initialValue: JsonEditorValue | null | undefined,
  defaultValue: JsonEditorValue | undefined,
): JsonEditorValue | null {
  if (initialValue !== undefined) return initialValue
  if (defaultValue !== undefined) return defaultValue
  return null
}

function makeUniqueObjectKey(source: Record<string, JsonEditorValue>, base = 'field'): string {
  if (!(base in source)) return base
  let index = 2
  while (`${base}_${index}` in source) index += 1
  return `${base}_${index}`
}

function JsonNodeEditor({
  value,
  onChange,
  depth = 0,
}: {
  value: JsonEditorValue
  onChange: (nextValue: JsonEditorValue) => void
  depth?: number
}) {
  const valueType = getJsonValueType(value)

  const handleTypeChange = (nextType: JsonValueType) => {
    if (nextType === valueType) return
    onChange(createDefaultValue(nextType))
  }

  return (
    <div className={cn('space-y-2 rounded-md border border-input/60 p-3', depth > 0 && 'bg-secondary/20')}>
      <div className="flex items-center gap-2">
        <label className="text-xs text-muted-foreground">Type</label>
        <select
          value={valueType}
          onChange={(event) => handleTypeChange(event.target.value as JsonValueType)}
          className="h-8 rounded-md border border-input bg-transparent px-2 text-xs"
        >
          <option value="object">object</option>
          <option value="array">array</option>
          <option value="string">string</option>
          <option value="number">number</option>
          <option value="boolean">boolean</option>
          <option value="null">null</option>
        </select>
      </div>

      {valueType === 'object' && (
        <div className="space-y-2">
          {Object.entries(value as Record<string, JsonEditorValue>).map(([key, childValue]) => (
            <div key={key} className="space-y-2 rounded-md border border-dashed border-input/70 p-2">
              <div className="flex items-center gap-2">
                <Input
                  value={key}
                  onChange={(event) => {
                    const nextKey = event.target.value.trim()
                    if (!nextKey || nextKey === key) return
                    const current = value as Record<string, JsonEditorValue>
                    const next: Record<string, JsonEditorValue> = {}
                    Object.entries(current).forEach(([existingKey, existingValue]) => {
                      next[existingKey === key ? nextKey : existingKey] = existingValue
                    })
                    onChange(next)
                  }}
                  className="h-8"
                  placeholder="field_name"
                />
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const next = { ...(value as Record<string, JsonEditorValue>) }
                    delete next[key]
                    onChange(next)
                  }}
                >
                  Remove
                </Button>
              </div>
              <JsonNodeEditor
                value={childValue}
                onChange={(nextChild) => {
                  onChange({
                    ...(value as Record<string, JsonEditorValue>),
                    [key]: nextChild,
                  })
                }}
                depth={depth + 1}
              />
            </div>
          ))}
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              const current = value as Record<string, JsonEditorValue>
              const nextKey = makeUniqueObjectKey(current)
              onChange({
                ...current,
                [nextKey]: '',
              })
            }}
          >
            Add Field
          </Button>
        </div>
      )}

      {valueType === 'array' && (
        <div className="space-y-2">
          {(value as JsonEditorValue[]).map((childValue, index) => (
            <div key={index} className="space-y-2 rounded-md border border-dashed border-input/70 p-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Item {index + 1}</span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const next = [...(value as JsonEditorValue[])]
                    next.splice(index, 1)
                    onChange(next)
                  }}
                >
                  Remove
                </Button>
              </div>
              <JsonNodeEditor
                value={childValue}
                onChange={(nextChild) => {
                  const next = [...(value as JsonEditorValue[])]
                  next[index] = nextChild
                  onChange(next)
                }}
                depth={depth + 1}
              />
            </div>
          ))}
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              onChange([...(value as JsonEditorValue[]), ''])
            }}
          >
            Add Item
          </Button>
        </div>
      )}

      {valueType === 'string' && (
        <Input
          value={String(value)}
          onChange={(event) => onChange(event.target.value)}
          placeholder="value"
        />
      )}

      {valueType === 'number' && (
        <Input
          type="number"
          value={String(value)}
          onChange={(event) => onChange(event.target.value === '' ? 0 : Number(event.target.value))}
        />
      )}

      {valueType === 'boolean' && (
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(event) => onChange(event.target.checked)}
            className="rounded border-input"
          />
          <span>{value ? 'true' : 'false'}</span>
        </label>
      )}

      {valueType === 'null' && (
        <p className="text-xs text-muted-foreground">Value is null.</p>
      )}
    </div>
  )
}

export function JsonEditorField({
  label,
  description,
  placeholder,
  initialValue,
  defaultValue = {},
  allowEmpty = true,
  onChange,
  className,
}: JsonEditorFieldProps) {
  const initial = useMemo(
    () => normalizeInitialValue(initialValue, defaultValue),
    [initialValue, defaultValue],
  )
  const [mode, setMode] = useState<JsonEditorMode>('structured')
  const [structuredValue, setStructuredValue] = useState<JsonEditorValue | null>(initial)
  const [rawValue, setRawValue] = useState(
    initial === null ? '' : JSON.stringify(initial, null, 2),
  )
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const isValid = error === null && (allowEmpty || rawValue.trim() !== '')
    onChange?.({
      parsed: structuredValue,
      raw: rawValue,
      isValid,
      error,
    })
  }, [allowEmpty, error, onChange, rawValue, structuredValue])

  const applyStructuredValue = (nextValue: JsonEditorValue) => {
    const nextRaw = JSON.stringify(nextValue, null, 2)
    setStructuredValue(nextValue)
    setRawValue(nextRaw)
    setError(null)
  }

  const handleRawChange = (nextRaw: string) => {
    setRawValue(nextRaw)
    if (!nextRaw.trim()) {
      setStructuredValue(null)
      setError(allowEmpty ? null : 'JSON is required')
      return
    }
    try {
      const parsed = JSON.parse(nextRaw) as JsonEditorValue
      setStructuredValue(parsed)
      setError(null)
    } catch {
      setError('Invalid JSON')
    }
  }

  const switchToStructured = () => {
    if (structuredValue === null) {
      applyStructuredValue(defaultValue)
    }
    setMode('structured')
  }

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between gap-3">
        <div className="space-y-1">
          <label className="text-sm font-medium">{label}</label>
          {description && <p className="text-xs text-muted-foreground">{description}</p>}
        </div>
        <div className="flex items-center gap-1 rounded-md border border-input p-1">
          <button
            type="button"
            onClick={switchToStructured}
            className={cn(
              'rounded px-2 py-1 text-xs transition-colors',
              mode === 'structured' ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground',
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

      {mode === 'structured' ? (
        <JsonNodeEditor
          value={structuredValue ?? defaultValue}
          onChange={applyStructuredValue}
        />
      ) : (
        <textarea
          value={rawValue}
          onChange={(event) => handleRawChange(event.target.value)}
          placeholder={placeholder}
          rows={6}
          className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono resize-y"
        />
      )}

      <div className="flex items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">
          Structured mode covers common edits. JSON mode remains available for copy/paste and exact control.
        </p>
        {error && <p className="text-xs text-red-400">{error}</p>}
      </div>
    </div>
  )
}
