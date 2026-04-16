import { useEffect, useMemo, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { APIError, mcpCall } from '@/api/client'
import { useCompositionProfiles, useSchemaAdapters } from '@/api/hooks/useCompositionProfiles'
import { useToolRegistry } from '@/api/hooks/useToolRegistry'
import type {
  CanonicalSchemaKind,
  CompositionComposeResult,
  CompositionProfile,
  CompositionPreviewResult,
  CompositionValidationResult,
} from '@/api/types'
import { BrowsePathField } from '@/components/evaluation/BrowsePathField'
import { DocumentPathListInput } from '@/components/pipeline/DocumentPathListInput'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { buildDatasetOutputPath, getDefaultDatasetOutputDir } from '@/lib/dataset-output'
import { FileStack, Loader2, ShieldCheck, SlidersHorizontal, Sparkles } from 'lucide-react'
import { toast } from 'sonner'

const REQUIRED_COMPOSITION_TOOLS = [
  'generate.list_profiles',
  'generate.preview_composition',
  'generate.compose_profiled_dataset',
  'generate.list_schema_adapters',
] as const

const OBJECTIVE_TO_SCHEMA_KIND: Record<string, CanonicalSchemaKind> = {
  sft: 'text_sft',
  vlm_sft: 'text_sft',
  dpo: 'preference_pair',
  grpo: 'reward_group',
  kto: 'binary_label',
}

function splitSourcePaths(value: string): string[] {
  return value
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean)
}

function titleCaseCapability(value: string): string {
  return value
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function parsePositiveInteger(value: string, label: string): number {
  const trimmed = value.trim()
  const parsed = Number.parseInt(trimmed, 10)
  if (!trimmed || !Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`)
  }
  return parsed
}

function parseCapabilityOverrides(
  profile: CompositionProfile,
  capabilityWeights: Record<string, string>,
): Record<string, number> {
  const overrides: Record<string, number> = {}

  for (const target of profile.capability_targets) {
    const rawValue = capabilityWeights[target.capability] ?? String(target.weight_percent)
    const trimmed = rawValue.trim()
    const parsed = Number.parseInt(trimmed, 10)
    if (!trimmed || !Number.isFinite(parsed) || parsed < 0) {
      throw new Error(`${titleCaseCapability(target.capability)} weight must be a non-negative integer`)
    }
    overrides[target.capability] = parsed
  }

  return overrides
}

function sumCapabilityWeights(
  profile: CompositionProfile | null,
  capabilityWeights: Record<string, string>,
): number | null {
  if (!profile) return null

  let total = 0
  for (const target of profile.capability_targets) {
    const rawValue = capabilityWeights[target.capability] ?? String(target.weight_percent)
    const parsed = Number.parseInt(rawValue.trim(), 10)
    if (!Number.isFinite(parsed) || parsed < 0) {
      return null
    }
    total += parsed
  }
  return total
}

function buildProfiledOutputPath(sourcePaths: string[], profileName: string): string {
  const sourceHint = sourcePaths[0] || profileName || 'profiled_dataset'
  return buildDatasetOutputPath(sourceHint, `${profileName || 'profiled'}_profiled`)
}

function getApiErrorPayload<T>(error: unknown): T | null {
  if (!(error instanceof APIError)) {
    return null
  }
  if (!error.details || typeof error.details !== 'object' || Array.isArray(error.details)) {
    return null
  }
  return error.details as T
}

function statusVariant(status: CompositionValidationResult['status']) {
  if (status === 'pass') return 'success'
  if (status === 'warn') return 'warning'
  return 'error'
}

export function ProfileComposer() {
  const queryClient = useQueryClient()
  const autoOutputPathRef = useRef<string | null>(null)
  const { data: tools = [], isLoading: toolsLoading } = useToolRegistry()
  const toolNames = useMemo(() => new Set(tools.map((tool) => tool.name)), [tools])
  const supportsComposer = REQUIRED_COMPOSITION_TOOLS.every((toolName) => toolNames.has(toolName))
  const supportsValidation = toolNames.has('validate.composition')

  const { data: profiles = [], isLoading: profilesLoading } = useCompositionProfiles(supportsComposer)
  const [profileName, setProfileName] = useState('')
  const selectedProfile = profiles.find((profile) => profile.name === profileName) ?? profiles[0] ?? null
  const [sourcePathsText, setSourcePathsText] = useState('')
  const sourcePaths = useMemo(() => splitSourcePaths(sourcePathsText), [sourcePathsText])
  const [objective, setObjective] = useState('')
  const expectedSchemaKind = objective ? OBJECTIVE_TO_SCHEMA_KIND[objective] : undefined
  const { data: schemaAdapters = [], isLoading: adaptersLoading } = useSchemaAdapters(
    expectedSchemaKind,
    supportsComposer && !!expectedSchemaKind,
  )
  const [schemaAdapterName, setSchemaAdapterName] = useState('')
  const [rowTarget, setRowTarget] = useState('200')
  const [outputPath, setOutputPath] = useState('')
  const [capabilityWeights, setCapabilityWeights] = useState<Record<string, string>>({})
  const [previewResult, setPreviewResult] = useState<CompositionPreviewResult | null>(null)
  const [composeResult, setComposeResult] = useState<CompositionComposeResult | null>(null)
  const [validationResult, setValidationResult] = useState<CompositionValidationResult | null>(null)
  const [activeAction, setActiveAction] = useState<'preview' | 'compose' | 'validate' | null>(null)

  useEffect(() => {
    if (!profileName && profiles[0]) {
      setProfileName(profiles[0].name)
    }
  }, [profileName, profiles])

  useEffect(() => {
    if (!selectedProfile) return

    setObjective((current) =>
      selectedProfile.allowed_objectives.includes(current)
        ? current
        : selectedProfile.default_objective,
    )
    setCapabilityWeights(
      Object.fromEntries(
        selectedProfile.capability_targets.map((target) => [
          target.capability,
          String(target.weight_percent),
        ]),
      ),
    )
    setPreviewResult(null)
    setComposeResult(null)
    setValidationResult(null)
  }, [selectedProfile?.name])

  useEffect(() => {
    if (schemaAdapterName && !schemaAdapters.some((adapter) => adapter.name === schemaAdapterName)) {
      setSchemaAdapterName('')
    }
  }, [schemaAdapterName, schemaAdapters])

  useEffect(() => {
    const nextAutoPath = buildProfiledOutputPath(sourcePaths, selectedProfile?.name ?? 'profiled_dataset')
    setOutputPath((current) => {
      const trimmedCurrent = current.trim()
      const previousAuto = autoOutputPathRef.current?.trim() ?? ''
      const shouldReplace = !trimmedCurrent || trimmedCurrent === previousAuto
      if (!shouldReplace || trimmedCurrent === nextAutoPath) {
        return current
      }
      autoOutputPathRef.current = nextAutoPath
      return nextAutoPath
    })
  }, [selectedProfile?.name, sourcePaths])

  const weightSum = sumCapabilityWeights(selectedProfile, capabilityWeights)
  const canPreview = !!selectedProfile && sourcePaths.length > 0 && !toolsLoading && !profilesLoading
  const composeUnavailableReason = selectedProfile
    ? !['general', 'coding', 'agent'].includes(selectedProfile.mode)
      ? 'Profiled generation is currently implemented only for general, coding, and agent profiles.'
      : !['sft', 'dpo', 'grpo', 'kto'].includes(objective)
        ? "Profiled generation currently supports only objective 'sft', 'dpo', 'grpo', or 'kto'."
        : null
    : 'Select a composition profile first.'
  const canCompose = canPreview && !composeUnavailableReason && !!outputPath.trim()

  async function buildRequestArgs(includeOutputPath: boolean) {
    if (!selectedProfile) {
      throw new Error('Select a composition profile first')
    }

    const rowTargetValue = parsePositiveInteger(rowTarget, 'Row target')
    const capabilityOverrides = parseCapabilityOverrides(selectedProfile, capabilityWeights)
    const args: Record<string, unknown> = {
      profile_name: selectedProfile.name,
      source_paths: sourcePaths,
      row_target: rowTargetValue,
      objective: objective || selectedProfile.default_objective,
      capability_overrides: capabilityOverrides,
    }

    if (schemaAdapterName) {
      args.schema_adapter_name = schemaAdapterName
    }
    if (includeOutputPath) {
      if (!outputPath.trim()) {
        throw new Error('Output path is required for profiled dataset generation')
      }
      args.output_path = outputPath.trim()
      args.format = 'jsonl'
    }

    return args
  }

  async function handlePreview() {
    try {
      setActiveAction('preview')
      setComposeResult(null)
      setValidationResult(null)
      const result = await mcpCall<CompositionPreviewResult>(
        'generate.preview_composition',
        await buildRequestArgs(false),
      )
      setPreviewResult(result)
      if (result.warnings.length > 0) {
        toast.warning(`Preview ready with ${result.warnings.length} warning${result.warnings.length === 1 ? '' : 's'}`)
      } else {
        toast.success('Composition preview is ready')
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Composition preview failed')
    } finally {
      setActiveAction(null)
    }
  }

  async function runValidation(datasetPath: string, manifestPath?: string) {
    try {
      setActiveAction('validate')
      const args: Record<string, unknown> = { dataset_path: datasetPath }
      if (manifestPath) {
        args.manifest_path = manifestPath
      }
      const result = await mcpCall<CompositionValidationResult>('validate.composition', args)
      setValidationResult(result)
      if (result.status === 'pass') {
        toast.success('Composition manifest validation passed')
      } else {
        toast.warning('Composition validation completed with warnings')
      }
    } catch (error) {
      const payload = getApiErrorPayload<CompositionValidationResult>(error)
      if (payload) {
        setValidationResult(payload)
      }
      toast.error(error instanceof Error ? error.message : 'Composition validation failed')
    } finally {
      setActiveAction(null)
    }
  }

  async function handleCompose() {
    try {
      setActiveAction('compose')
      setValidationResult(null)
      const result = await mcpCall<CompositionComposeResult>(
        'generate.compose_profiled_dataset',
        await buildRequestArgs(true),
      )
      setComposeResult(result)
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success(`Profiled dataset saved to ${result.dataset.file_path}`)

      if (supportsValidation) {
        await runValidation(result.dataset.file_path, result.manifest_path)
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Profiled dataset generation failed')
    } finally {
      setActiveAction(null)
    }
  }

  if (!supportsComposer && !toolsLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <SlidersHorizontal className="h-4 w-4" />
            Profile Composer
          </CardTitle>
          <CardDescription>
            This gateway does not advertise the profiled dataset composition tools yet.
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <SlidersHorizontal className="h-4 w-4" />
          Profile Composer
        </CardTitle>
        <CardDescription>
          Preview and generate profile-driven datasets with editable capability weights and schema adapters.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">Preview: all advertised profiles</Badge>
          <Badge variant="outline">Generate: general + coding + agent with sft only for now</Badge>
          <Badge variant="outline">Default output root: {getDefaultDatasetOutputDir()}</Badge>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Composition Profile</label>
            <select
              value={selectedProfile?.name ?? ''}
              onChange={(event) => setProfileName(event.target.value)}
              className="w-full h-9 rounded-md border border-input bg-transparent px-3 text-sm text-foreground"
              disabled={profilesLoading}
            >
              <option value="">Select profile...</option>
              {profiles.map((profile) => (
                <option key={profile.name} value={profile.name}>
                  {profile.name} ({profile.mode})
                </option>
              ))}
            </select>
            {selectedProfile ? (
              <p className="text-xs text-muted-foreground">{selectedProfile.description}</p>
            ) : null}
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Objective</label>
            <select
              value={objective}
              onChange={(event) => setObjective(event.target.value)}
              className="w-full h-9 rounded-md border border-input bg-transparent px-3 text-sm text-foreground"
              disabled={!selectedProfile}
            >
              <option value="">Select objective...</option>
              {selectedProfile?.allowed_objectives.map((allowedObjective) => (
                <option key={allowedObjective} value={allowedObjective}>
                  {allowedObjective}
                </option>
              ))}
            </select>
            {composeUnavailableReason ? (
              <p className="text-xs text-muted-foreground">{composeUnavailableReason}</p>
            ) : (
              <p className="text-xs text-muted-foreground">This combination can be previewed and generated.</p>
            )}
          </div>
        </div>

        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">Source Paths</label>
          <DocumentPathListInput
            value={sourcePathsText}
            onChange={setSourcePathsText}
            disabled={activeAction !== null}
            helperText="Upload one or more source files or folders. Each line is sent to profiled preview and generation as a source path."
          />
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">Row Target</label>
            <Input
              type="number"
              min="1"
              value={rowTarget}
              onChange={(event) => setRowTarget(event.target.value)}
              placeholder="200"
            />
          </div>

          <div className="space-y-1 lg:col-span-2">
            <label className="text-xs font-medium text-muted-foreground">Schema Adapter</label>
            <select
              value={schemaAdapterName}
              onChange={(event) => setSchemaAdapterName(event.target.value)}
              className="w-full h-9 rounded-md border border-input bg-transparent px-3 text-sm text-foreground"
              disabled={!expectedSchemaKind || adaptersLoading}
            >
              <option value="">Default for {objective || 'objective'}</option>
              {schemaAdapters.map((adapter) => (
                <option key={adapter.name} value={adapter.name}>
                  {adapter.name}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              Standard adapters are listed here. Runtime adapter registration still lives in the Tools page.
            </p>
          </div>
        </div>

        <div className="space-y-2 rounded-md border border-border/60 bg-secondary/20 p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-medium">Capability Mix</h3>
              <p className="text-xs text-muted-foreground">
                MCP Tuna will normalize the mix if it does not sum to 100. Keep weights grounded in source coverage.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant={weightSum === 100 ? 'success' : 'warning'}>
                Weight Sum: {weightSum ?? 'invalid'}
              </Badge>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => {
                  if (!selectedProfile) return
                  setCapabilityWeights(
                    Object.fromEntries(
                      selectedProfile.capability_targets.map((target) => [
                        target.capability,
                        String(target.weight_percent),
                      ]),
                    ),
                  )
                }}
                disabled={!selectedProfile}
              >
                Reset Defaults
              </Button>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {selectedProfile?.capability_targets.map((target) => (
              <div key={target.capability} className="space-y-1 rounded-md border border-border/50 bg-background/60 p-3">
                <label className="text-xs font-medium text-muted-foreground">
                  {titleCaseCapability(target.capability)}
                </label>
                <Input
                  type="number"
                  min="0"
                  value={capabilityWeights[target.capability] ?? String(target.weight_percent)}
                  onChange={(event) =>
                    setCapabilityWeights((current) => ({
                      ...current,
                      [target.capability]: event.target.value,
                    }))
                  }
                />
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">Output Path</label>
          <BrowsePathField
            value={outputPath}
            onChange={setOutputPath}
            disabled={activeAction !== null}
            placeholder={`${getDefaultDatasetOutputDir()}/general_instruction_profiled.jsonl`}
            helperText="Use a dataset filename under a server-visible directory. Browse appends or replaces the filename safely."
            preferredRootIds={['data', 'output', 'workspace', 'uploads', 'hf_cache']}
            directorySelectionMode="append-filename"
            defaultFileName={outputPath.split(/[\\/]/).filter(Boolean).at(-1) ?? 'profiled_dataset.jsonl'}
          />
        </div>

        <div className="flex flex-wrap gap-2">
          <Button onClick={handlePreview} disabled={!canPreview || activeAction !== null}>
            {activeAction === 'preview' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            {activeAction === 'preview' ? 'Previewing...' : 'Preview Composition'}
          </Button>
          <Button
            onClick={handleCompose}
            disabled={!canCompose || activeAction !== null}
            variant="secondary"
          >
            {activeAction === 'compose' ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileStack className="h-4 w-4" />}
            {activeAction === 'compose' ? 'Generating...' : 'Generate Dataset'}
          </Button>
          {supportsValidation && composeResult?.dataset.file_path ? (
            <Button
              type="button"
              variant="outline"
              onClick={() => runValidation(composeResult.dataset.file_path, composeResult.manifest_path)}
              disabled={activeAction !== null}
            >
              {activeAction === 'validate' ? <Loader2 className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
              {activeAction === 'validate' ? 'Validating...' : 'Validate Composition'}
            </Button>
          ) : null}
        </div>

        {previewResult ? (
          <div className="space-y-3 rounded-md border border-border/60 bg-secondary/20 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary">Preview: {previewResult.profile_name}</Badge>
              <Badge variant="outline">Objective: {previewResult.objective}</Badge>
              <Badge variant="outline">Chunks: {previewResult.source_totals.estimated_chunks}</Badge>
              <Badge variant="outline">Adapter: {previewResult.schema_adapter.name}</Badge>
            </div>
            <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
              {Object.entries(previewResult.row_plan).map(([capability, plannedRows]) => (
                <div key={capability} className="rounded-md border border-border/50 bg-background/60 p-3 text-sm">
                  <div className="font-medium">{titleCaseCapability(capability)}</div>
                  <div className="mt-1 text-muted-foreground">
                    {plannedRows} rows
                    {' '}({previewResult.resolved_mix[capability] ?? 0}%)
                  </div>
                </div>
              ))}
            </div>
            {previewResult.warnings.length > 0 ? (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Warnings</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  {previewResult.warnings.map((warning) => (
                    <li key={warning}>• {warning}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        ) : null}

        {composeResult ? (
          <div className="space-y-3 rounded-md border border-border/60 bg-secondary/20 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="success">Generated: {composeResult.row_count} rows</Badge>
              <Badge variant="outline">Dataset: {composeResult.dataset.file_path.split(/[\\/]/).pop()}</Badge>
              <Badge variant="outline">Manifest: {composeResult.manifest_path.split(/[\\/]/).pop()}</Badge>
            </div>
            <p className="text-xs text-muted-foreground break-all">{composeResult.dataset.file_path}</p>
            {composeResult.warnings.length > 0 ? (
              <ul className="space-y-1 text-sm text-muted-foreground">
                {composeResult.warnings.map((warning) => (
                  <li key={warning}>• {warning}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}

        {validationResult ? (
          <div className="space-y-3 rounded-md border border-border/60 bg-secondary/20 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={statusVariant(validationResult.status)}>
                Validation: {validationResult.status}
              </Badge>
              <Badge variant="outline">Rows: {validationResult.row_count}</Badge>
              <Badge variant="outline">
                Source refs: {Math.round(validationResult.source_ref_coverage * 100)}%
              </Badge>
            </div>
            {validationResult.errors.length > 0 ? (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Errors</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  {validationResult.errors.map((error) => (
                    <li key={error}>• {error}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {validationResult.warnings.length > 0 ? (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Warnings</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  {validationResult.warnings.map((warning) => (
                    <li key={warning}>• {warning}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
