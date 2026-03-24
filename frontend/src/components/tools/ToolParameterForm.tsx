import { useState, useCallback, useEffect, useRef } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { JSONSchemaProperty } from '@/api/types'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { BrowsePathField } from '@/components/evaluation/BrowsePathField'
import { JsonEditorField } from '@/components/shared/JsonEditorField'
import { ModelPathField } from '@/components/pipeline/ModelPathField'
import { buildToolExecutionOutputDir } from '@/lib/training-capabilities'
import { toast } from 'sonner'

interface ToolParameterFormProps {
  toolName: string
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

function getDefaultFormValues(schema: ToolParameterFormProps['schema']): Record<string, unknown> {
  const defaults: Record<string, unknown> = {}

  for (const [name, prop] of Object.entries(schema.properties ?? {})) {
    if (prop.default !== undefined) {
      defaults[name] = prop.default
    }
  }

  return defaults
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
  onJsonValidityChange,
}: {
  name: string
  schema: JSONSchemaProperty
  required: boolean
  value: unknown
  onChange: (val: unknown) => void
  onJsonValidityChange: (name: string, isValid: boolean) => void
}) {
  const stringDefault = typeof schema.default === 'string' ? schema.default : undefined
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
          placeholder={stringDefault || 'meta-llama/Llama-3.2-3B-Instruct'}
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
          placeholder={stringDefault || '/path/to/adapter'}
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
          placeholder={stringDefault || `/${normalizedName}`}
          helperText={helperText}
          allowFiles={!isDirectoryOnly}
          allowDirectories={true}
          preferredRootIds={normalizedName.startsWith('output_') || normalizedName === 'output_dir'
            ? ['output', 'workspace', 'uploads', 'hf_cache']
            : ['workspace', 'output', 'uploads', 'hf_cache']}
          directorySelectionMode={isOutputPath ? 'append-filename' : 'replace'}
          defaultFileName={isOutputPath ? inferDefaultFileName(value, stringDefault ?? '') : undefined}
        />
      </div>
    )
  }

  if (schema.type === 'object' || schema.type === 'array') {
    const isMessagesField = normalizedName === 'messages'
    return (
      <div className="space-y-1">
        <JsonEditorField
          label={name}
          description={[
            schema.description,
            isMessagesField
              ? 'Use canonical multimodal message blocks. Upload images first, then reference the returned image_path.'
              : null,
          ].filter(Boolean).join(' ')}
          initialValue={typeof value === 'string' ? null : (value as never)}
          defaultValue={schema.type === 'array' ? [] : {}}
          placeholder={stringDefault || getJsonPlaceholder(name, schema)}
          allowEmpty={!required}
          onChange={({ parsed, isValid }) => {
            onJsonValidityChange(name, isValid)
            if (parsed === null) {
              onChange(undefined)
              return
            }
            onChange(parsed)
          }}
          className="pt-1"
        />
        <div className="flex items-center gap-2 text-xs">
          <Badge variant="outline" className="text-[10px] py-0">{schema.type}</Badge>
          {required && <Badge variant="destructive" className="text-[10px] py-0">required</Badge>}
        </div>
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
        step={schema.type === 'integer' ? '1' : 'any'}
      />
    </div>
  )
}

export function ToolParameterForm({ toolName, schema, onSubmit, isLoading }: ToolParameterFormProps) {
  const [values, setValues] = useState<Record<string, unknown>>(() => getDefaultFormValues(schema))
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [jsonValidity, setJsonValidity] = useState<Record<string, boolean>>({})
  const autoOutputDirRef = useRef<string | null>(null)

  const required = new Set(schema.required ?? [])
  const entries = Object.entries(schema.properties ?? {})
  const requiredFields = entries.filter(([k]) => required.has(k))
  const optionalFields = entries.filter(([k]) => !required.has(k))

  useEffect(() => {
    if (!schema.properties.output_dir) return

    const autoOutputDir = buildToolExecutionOutputDir(toolName, values)
    if (!autoOutputDir) return

    setValues((prev) => {
      const currentOutputDir = typeof prev.output_dir === 'string' ? prev.output_dir.trim() : ''
      const lastAutoOutputDir = autoOutputDirRef.current?.trim() ?? ''
      const shouldReplace = !currentOutputDir || currentOutputDir === lastAutoOutputDir

      if (!shouldReplace || currentOutputDir === autoOutputDir) {
        return prev
      }

      autoOutputDirRef.current = autoOutputDir
      return {
        ...prev,
        output_dir: autoOutputDir,
      }
    })
  }, [schema.properties, toolName, values])

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

  const handleJsonValidityChange = useCallback((name: string, isValid: boolean) => {
    setJsonValidity((prev) => ({
      ...prev,
      [name]: isValid,
    }))
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const invalidJsonFields = Object.entries(jsonValidity)
      .filter(([, isValid]) => !isValid)
      .map(([name]) => name)
    if (invalidJsonFields.length > 0) {
      toast.error(`Invalid JSON in: ${invalidJsonFields.join(', ')}`)
      return
    }
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
              onJsonValidityChange={handleJsonValidityChange}
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
                  onJsonValidityChange={handleJsonValidityChange}
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
