import { useState, useCallback, useEffect, useRef } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { JSONSchemaProperty } from '@/api/types'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { BrowsePathField } from '@/components/evaluation/BrowsePathField'
import { JsonEditorField, type JsonEditorValue } from '@/components/shared/JsonEditorField'
import { PreferenceDatasetAnalysisCard } from '@/components/shared/PreferenceDatasetAnalysisCard'
import { ModelPathField } from '@/components/pipeline/ModelPathField'
import {
  buildToolExecutionOutputDir,
  extractPreferenceStartingRecipePatch,
  resolvePreferenceTechniqueFromTool,
} from '@/lib/training-capabilities'
import { describeInitialAdapterRecipeHint } from '@/lib/training-copy'
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

const SCHEMA_ADAPTER_FIELD_MAP_DEFAULTS: Record<string, Record<string, string>> = {
  text_sft: {
    instruction: 'instruction',
    input: 'input',
    output: 'output',
  },
  preference_pair: {
    prompt: 'prompt',
    chosen: 'chosen',
    rejected: 'rejected',
  },
  reward_group: {
    prompt: 'prompt',
    responses: 'responses',
    rewards: 'rewards',
  },
  binary_label: {
    prompt: 'prompt',
    completion: 'completion',
    label: 'label',
  },
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

function cloneJsonValue(value: JsonEditorValue): JsonEditorValue {
  if (Array.isArray(value)) {
    return value.map(cloneJsonValue)
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [key, cloneJsonValue(child)]),
    )
  }
  return value
}

function getSchemaAdapterFieldMapDefault(canonicalKind: unknown): Record<string, string> {
  const key = typeof canonicalKind === 'string' ? canonicalKind : 'text_sft'
  return {
    ...(SCHEMA_ADAPTER_FIELD_MAP_DEFAULTS[key] ?? SCHEMA_ADAPTER_FIELD_MAP_DEFAULTS.text_sft),
  }
}

function areStringMapsEqual(left: unknown, right: unknown): boolean {
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object') {
    return false
  }

  const leftEntries = Object.entries(left as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b))
  const rightEntries = Object.entries(right as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b))
  if (leftEntries.length !== rightEntries.length) return false

  return leftEntries.every(([leftKey, leftValue], index) => {
    const [rightKey, rightValue] = rightEntries[index]
    return leftKey === rightKey && leftValue === rightValue
  })
}

function getJsonFieldDefaultValue(
  toolName: string,
  name: string,
  schema: JSONSchemaProperty,
  values: Record<string, unknown>,
): JsonEditorValue {
  if (toolName === 'generate.register_schema_adapter' && name === 'field_map') {
    return getSchemaAdapterFieldMapDefault(values.canonical_kind ?? schema.default)
  }
  if (schema.default !== undefined) {
    return cloneJsonValue(schema.default as JsonEditorValue)
  }
  return schema.type === 'array' ? [] : {}
}

function getJsonFieldDescription(
  toolName: string,
  name: string,
  schema: JSONSchemaProperty,
  values: Record<string, unknown>,
): string {
  const notes: string[] = []
  if (schema.description) {
    notes.push(schema.description)
  }
  if (name.toLowerCase() === 'messages') {
    notes.push(
      'Use canonical multimodal message blocks. Upload images first, then reference the returned image_path.',
    )
  }
  if (toolName === 'generate.register_schema_adapter' && name === 'field_map') {
    const canonicalKind = typeof values.canonical_kind === 'string' ? values.canonical_kind : 'text_sft'
    notes.push(
      `Keys are canonical fields and values are your dataset column names. Seeded with the standard ${canonicalKind} schema.`,
    )
  }
  return notes.join(' ')
}

function getDefaultFormValues(
  toolName: string,
  schema: ToolParameterFormProps['schema'],
): Record<string, unknown> {
  const defaults: Record<string, unknown> = {}

  for (const [name, prop] of Object.entries(schema.properties ?? {})) {
    if (prop.default !== undefined) {
      defaults[name] = prop.default
    }
  }

  if (toolName === 'generate.register_schema_adapter') {
    if (defaults.canonical_kind === undefined) {
      defaults.canonical_kind = 'text_sft'
    }
    if (defaults.field_map === undefined && schema.properties.field_map?.type === 'object') {
      defaults.field_map = getSchemaAdapterFieldMapDefault(defaults.canonical_kind)
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
  toolName,
  name,
  schema,
  required,
  value,
  allValues,
  onChange,
  onJsonValidityChange,
}: {
  toolName: string
  name: string
  schema: JSONSchemaProperty
  required: boolean
  value: unknown
  allValues: Record<string, unknown>
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
    const jsonDefaultValue = getJsonFieldDefaultValue(toolName, name, schema, allValues)
    return (
      <div className="space-y-1">
        <JsonEditorField
          label={name}
          description={getJsonFieldDescription(toolName, name, schema, allValues)}
          initialValue={typeof value === 'string' ? null : (value as never)}
          defaultValue={jsonDefaultValue}
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
  const [values, setValues] = useState<Record<string, unknown>>(() => getDefaultFormValues(toolName, schema))
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [jsonValidity, setJsonValidity] = useState<Record<string, boolean>>({})
  const autoOutputDirRef = useRef<string | null>(null)

  const required = new Set(schema.required ?? [])
  const entries = Object.entries(schema.properties ?? {})
  const requiredFields = entries.filter(([k]) => required.has(k))
  const optionalFields = entries.filter(([k]) => !required.has(k))
  const datasetPath = typeof values.dataset_path === 'string' ? values.dataset_path : ''
  const preferenceTechnique = resolvePreferenceTechniqueFromTool(toolName, values.technique)

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

  useEffect(() => {
    if (toolName !== 'generate.register_schema_adapter' || !schema.properties.field_map) {
      return
    }

    const suggested = getSchemaAdapterFieldMapDefault(values.canonical_kind)
    setValues((prev) => {
      const current = prev.field_map
      if (current === undefined) {
        return {
          ...prev,
          field_map: suggested,
        }
      }

      const isAutoTemplate = Object.values(SCHEMA_ADAPTER_FIELD_MAP_DEFAULTS).some((template) =>
        areStringMapsEqual(current, template),
      )
      if (!isAutoTemplate || areStringMapsEqual(current, suggested)) {
        return prev
      }

      return {
        ...prev,
        field_map: suggested,
      }
    })
  }, [schema.properties.field_map, toolName, values.canonical_kind])

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

  const applyPreferenceStartingRecipe = useCallback((recipe: Record<string, string | number | boolean>) => {
    const patch = extractPreferenceStartingRecipePatch(recipe)

    setValues((prev) => {
      const next = { ...prev }
      if (patch.num_epochs !== undefined && schema.properties.num_epochs) {
        next.num_epochs = patch.num_epochs
      }
      if (patch.learning_rate !== undefined && schema.properties.learning_rate) {
        next.learning_rate = patch.learning_rate
      }
      if (schema.properties.auto_tune_defaults) {
        next.auto_tune_defaults = true
      }
      return next
    })

    if (
      patch.start_from_sft_checkpoint &&
      schema.properties.adapter_path &&
      !(typeof values.adapter_path === 'string' && values.adapter_path.trim())
    ) {
      setShowAdvanced(true)
      toast.info(describeInitialAdapterRecipeHint('adapter_path'))
    } else {
      toast.success('Applied the safe preference starting recipe')
    }
  }, [schema.properties, values.adapter_path])

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
              toolName={toolName}
              name={name}
              schema={prop}
              required={true}
              value={values[name]}
              allValues={values}
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
                  toolName={toolName}
                  name={name}
                  schema={prop}
                  required={false}
                  value={values[name]}
                  allValues={values}
                  onChange={(v) => handleChange(name, v)}
                  onJsonValidityChange={handleJsonValidityChange}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {preferenceTechnique && datasetPath.trim() && (
        <PreferenceDatasetAnalysisCard
          datasetPath={datasetPath}
          technique={preferenceTechnique}
          onApplyStartingRecipe={applyPreferenceStartingRecipe}
        />
      )}

      <Button type="submit" disabled={isLoading} className="w-full">
        {isLoading ? 'Executing...' : 'Execute Tool'}
      </Button>
    </form>
  )
}
