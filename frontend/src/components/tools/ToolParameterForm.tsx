import { useState, useCallback } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { JSONSchemaProperty } from '@/api/types'
import { ChevronDown, ChevronRight } from 'lucide-react'

interface ToolParameterFormProps {
  schema: {
    properties: Record<string, JSONSchemaProperty>
    required?: string[]
  }
  onSubmit: (args: Record<string, unknown>) => void
  isLoading?: boolean
}

function ParameterField({
  name,
  schema,
  required,
  value,
  onChange,
}: {
  name: string
  schema: JSONSchemaProperty
  required: boolean
  value: unknown
  onChange: (val: unknown) => void
}) {
  const placeholder = schema.default !== undefined ? `Default: ${JSON.stringify(schema.default)}` : ''

  if (schema.type === 'boolean') {
    return (
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={value === true || (value === undefined && schema.default === true)}
          onChange={(e) => onChange(e.target.checked)}
          className="rounded border-input"
        />
        <span className="text-sm">{name}</span>
        {required && <Badge variant="destructive" className="text-[10px] py-0">required</Badge>}
        {schema.description && (
          <span className="text-xs text-muted-foreground ml-1">{schema.description}</span>
        )}
      </label>
    )
  }

  if (schema.enum) {
    return (
      <div className="space-y-1">
        <label className="flex items-center gap-2 text-sm">
          {name}
          {required && <Badge variant="destructive" className="text-[10px] py-0">required</Badge>}
        </label>
        {schema.description && (
          <p className="text-xs text-muted-foreground">{schema.description}</p>
        )}
        <select
          value={String(value ?? schema.default ?? '')}
          onChange={(e) => onChange(e.target.value)}
          className="w-full h-9 rounded-md border border-input bg-transparent px-3 py-1 text-sm"
        >
          <option value="">Select...</option>
          {schema.enum.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      </div>
    )
  }

  if (schema.type === 'object' || schema.type === 'array') {
    return (
      <div className="space-y-1">
        <label className="flex items-center gap-2 text-sm">
          {name}
          <Badge variant="outline" className="text-[10px] py-0">{schema.type}</Badge>
          {required && <Badge variant="destructive" className="text-[10px] py-0">required</Badge>}
        </label>
        {schema.description && (
          <p className="text-xs text-muted-foreground">{schema.description}</p>
        )}
        <textarea
          value={typeof value === 'string' ? value : value ? JSON.stringify(value, null, 2) : ''}
          onChange={(e) => {
            try {
              onChange(JSON.parse(e.target.value))
            } catch {
              onChange(e.target.value)
            }
          }}
          placeholder={placeholder || `Enter ${schema.type} as JSON`}
          rows={3}
          className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono resize-y"
        />
      </div>
    )
  }

  return (
    <div className="space-y-1">
      <label className="flex items-center gap-2 text-sm">
        {name}
        {required && <Badge variant="destructive" className="text-[10px] py-0">required</Badge>}
        {schema.type === 'number' || schema.type === 'integer' ? (
          <Badge variant="outline" className="text-[10px] py-0">{schema.type}</Badge>
        ) : null}
      </label>
      {schema.description && (
        <p className="text-xs text-muted-foreground">{schema.description}</p>
      )}
      <Input
        type={schema.type === 'number' || schema.type === 'integer' ? 'number' : 'text'}
        value={String(value ?? '')}
        onChange={(e) => {
          const v = e.target.value
          if (schema.type === 'number') onChange(v === '' ? undefined : parseFloat(v))
          else if (schema.type === 'integer') onChange(v === '' ? undefined : parseInt(v, 10))
          else onChange(v || undefined)
        }}
        placeholder={placeholder}
        step={schema.type === 'integer' ? '1' : 'any'}
      />
    </div>
  )
}

export function ToolParameterForm({ schema, onSubmit, isLoading }: ToolParameterFormProps) {
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [showAdvanced, setShowAdvanced] = useState(false)

  const required = new Set(schema.required ?? [])
  const entries = Object.entries(schema.properties ?? {})
  const requiredFields = entries.filter(([k]) => required.has(k))
  const optionalFields = entries.filter(([k]) => !required.has(k))

  const handleChange = useCallback((name: string, val: unknown) => {
    setValues((prev) => {
      const next = { ...prev }
      if (val === undefined || val === '') {
        delete next[name]
      } else {
        next[name] = val
      }
      return next
    })
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const args: Record<string, unknown> = {}
    for (const [key, val] of Object.entries(values)) {
      if (val !== undefined && val !== '') {
        args[key] = val
      }
    }
    onSubmit(args)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {requiredFields.length > 0 && (
        <div className="space-y-3">
          {requiredFields.map(([name, prop]) => (
            <ParameterField
              key={name}
              name={name}
              schema={prop}
              required={true}
              value={values[name]}
              onChange={(v) => handleChange(name, v)}
            />
          ))}
        </div>
      )}

      {optionalFields.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {showAdvanced ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            {optionalFields.length} optional parameters
          </button>
          {showAdvanced && (
            <div className="space-y-3 mt-3 pl-3 border-l-2 border-border">
              {optionalFields.map(([name, prop]) => (
                <ParameterField
                  key={name}
                  name={name}
                  schema={prop}
                  required={false}
                  value={values[name]}
                  onChange={(v) => handleChange(name, v)}
                />
              ))}
            </div>
          )}
        </div>
      )}

      <Button type="submit" disabled={isLoading} className="w-full">
        {isLoading ? 'Executing...' : 'Execute Tool'}
      </Button>
    </form>
  )
}
