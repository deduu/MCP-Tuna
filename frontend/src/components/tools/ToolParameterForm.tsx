import { useState, useCallback } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { JSONSchemaProperty } from '@/api/types'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { BrowsePathField } from '@/components/evaluation/BrowsePathField'
import { ModelPathField } from '@/components/pipeline/ModelPathField'

interface ToolParameterFormProps {
  schema: {
    properties: Record<string, JSONSchemaProperty>
    required?: string[]
  }
  onSubmit: (args: Record<string, unknown>) => void
  isLoading?: boolean
}

const KNOWN_SELECT_OPTIONS: Record<string, string[]> = {
  technique: ['sft', 'vlm_sft', 'dpo', 'grpo', 'kto'],
  target_format: ['sft', 'dpo', 'grpo', 'kto'],
  difficulty_order: ['easy_first', 'hard_first'],
  use_case: ['general', 'low_memory', 'speed', 'quality', 'multilingual', 'indonesian'],
}

function getSelectOptions(name: string, schema: JSONSchemaProperty): string[] | null {
  if (schema.enum?.length) {
    return schema.enum.map((option) => String(option))
  }

  return KNOWN_SELECT_OPTIONS[name.toLowerCase()] ?? null
}

function isModelField(name: string, schema: JSONSchemaProperty): boolean {
  if (schema.type !== 'string') return false
  const normalized = name.toLowerCase()
  return normalized === 'model_name' || normalized === 'model_path' || normalized === 'base_model'
}

function isAdapterField(name: string, schema: JSONSchemaProperty): boolean {
  if (schema.type !== 'string') return false
  return name.toLowerCase() === 'adapter_path'
}

function isPathField(name: string, schema: JSONSchemaProperty): boolean {
  if (schema.type !== 'string') return false
  if (schema.format === 'path') return true
  if (isModelField(name, schema) || isAdapterField(name, schema)) return false

  const normalized = name.toLowerCase()
  return normalized.endsWith('_path') || normalized.endsWith('_dir')
}

function inferDefaultFileName(value: unknown, placeholder: string): string | undefined {
  const source = typeof value === 'string' && value.trim()
    ? value.trim()
    : placeholder.trim()

  if (!source) return undefined

  const normalized = source.replace(/[\\/]+$/, '')
  const lastSegment = normalized.split(/[\\/]/).filter(Boolean).at(-1)
  if (!lastSegment) return undefined
  if (!lastSegment.includes('.')) return undefined
  return lastSegment
}

function getJsonPlaceholder(name: string, schema: JSONSchemaProperty): string {
  if (name.toLowerCase() === 'messages') {
    return '[{"role":"user","content":[{"type":"image_path","image_path":"uploads/images/example.png"},{"type":"text","text":"Describe this image."}]}]'
  }
  return `Enter ${schema.type} as JSON`
}

function renderLabel(name: string, required: boolean, schema: JSONSchemaProperty) {
  return (
    <label className="flex items-center gap-2 text-sm">
      {name}
      {required && <Badge variant="destructive" className="text-[10px] py-0">required</Badge>}
      {schema.type === 'number' || schema.type === 'integer' ? (
        <Badge variant="outline" className="text-[10px] py-0">{schema.type}</Badge>
      ) : null}
    </label>
  )
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
  const defaultLabel = schema.default !== undefined ? `Default: ${JSON.stringify(schema.default)}` : ''
  const placeholder = typeof schema.default === 'string' ? schema.default : defaultLabel
  const selectOptions = getSelectOptions(name, schema)
  const normalizedName = name.toLowerCase()

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

  if (selectOptions) {
    return (
      <div className="space-y-1">
        {renderLabel(name, required, schema)}
        {schema.description && (
          <p className="text-xs text-muted-foreground">{schema.description}</p>
        )}
        <select
          value={String(value ?? schema.default ?? '')}
          onChange={(e) => onChange(e.target.value)}
          className="w-full h-9 rounded-md border border-input bg-transparent px-3 py-1 text-sm"
        >
          <option value="">Select...</option>
          {selectOptions.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      </div>
    )
  }

  if (isModelField(name, schema)) {
    return (
      <div className="space-y-1">
        {renderLabel(name, required, schema)}
        {schema.description && (
          <p className="text-xs text-muted-foreground">{schema.description}</p>
        )}
        <ModelPathField
          value={typeof value === 'string' ? value : ''}
          onChange={(nextValue) => onChange(nextValue || undefined)}
          placeholder={placeholder || 'meta-llama/Llama-3.2-3B-Instruct'}
        />
      </div>
    )
  }

  if (isAdapterField(name, schema)) {
    return (
      <div className="space-y-1">
        {renderLabel(name, required, schema)}
        {schema.description && (
          <p className="text-xs text-muted-foreground">{schema.description}</p>
        )}
        <ModelPathField
          value={typeof value === 'string' ? value : ''}
          onChange={(nextValue) => onChange(nextValue || undefined)}
          placeholder={placeholder || '/path/to/adapter'}
          validationPurpose="adapter"
        />
      </div>
    )
  }

  if (isPathField(name, schema)) {
    const isDirectoryOnly = normalizedName.endsWith('_dir')
    const isOutputPath = normalizedName === 'output_path'
    const helperText = isDirectoryOnly
      ? 'Browse a backend-visible folder or type a path directly.'
      : 'Browse backend-visible files and folders or type a path directly.'

    return (
      <div className="space-y-1">
        {renderLabel(name, required, schema)}
        {schema.description && (
          <p className="text-xs text-muted-foreground">{schema.description}</p>
        )}
        <BrowsePathField
          value={typeof value === 'string' ? value : ''}
          onChange={(nextValue) => onChange(nextValue || undefined)}
          placeholder={placeholder || `/${normalizedName}`}
          helperText={helperText}
          allowFiles={!isDirectoryOnly}
          allowDirectories={true}
          preferredRootIds={normalizedName.startsWith('output_') || normalizedName === 'output_dir'
            ? ['output', 'workspace', 'uploads', 'hf_cache']
            : ['workspace', 'output', 'uploads', 'hf_cache']}
          directorySelectionMode={isOutputPath ? 'append-filename' : 'replace'}
          defaultFileName={isOutputPath ? inferDefaultFileName(value, placeholder) : undefined}
        />
      </div>
    )
  }

  if (schema.type === 'object' || schema.type === 'array') {
    const isMessagesField = normalizedName === 'messages'
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
        {isMessagesField && (
          <p className="text-xs text-muted-foreground">
            Use canonical multimodal message blocks. Upload images first, then reference the returned `image_path`.
          </p>
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
          placeholder={placeholder || getJsonPlaceholder(name, schema)}
          rows={3}
          className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono resize-y"
        />
      </div>
    )
  }

  return (
    <div className="space-y-1">
      {renderLabel(name, required, schema)}
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
