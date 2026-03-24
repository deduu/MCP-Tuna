import { useEffect, useRef, useState } from 'react'
import { AlertCircle, CheckCircle2, ChevronDown, ChevronLeft, FolderSearch, Info, Loader2 } from 'lucide-react'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { useDeploymentBrowseDir, useDeploymentBrowseRoots } from '@/api/hooks/useTraining'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

type ValidationTone = 'success' | 'warning' | 'error' | 'info'

export interface ModelPathValidation {
  tone: ValidationTone
  message: string
  isAdapter: boolean
  isLocalPath: boolean
}

interface ModelPathFieldProps {
  value: string
  onChange: (value: string) => void
  disabled?: boolean
  placeholder?: string
  helperText?: string
  validationPurpose?: 'model' | 'adapter'
  onValidationChange?: (validation: ModelPathValidation | null) => void
}

function normalizeFsPath(value: string): string {
  return value.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase()
}

function preferredRootId(
  roots: Array<{ id: string; exists: boolean }>,
  purpose: 'model' | 'adapter',
): string {
  const preferredOrder = purpose === 'adapter'
    ? ['output', 'workspace', 'uploads', 'hf_cache']
    : ['hf_cache', 'workspace', 'output', 'uploads']

  for (const rootId of preferredOrder) {
    const match = roots.find((root) => root.id === rootId && root.exists)
    if (match) return match.id
  }

  return roots.find((root) => root.exists)?.id ?? roots[0]?.id ?? ''
}

function resolveValueToBrowseLocation(
  value: string,
  roots: Array<{ id: string; path: string; exists: boolean }>,
): { rootId: string; browsePath: string } | null {
  const normalizedValue = normalizeFsPath(value.trim())
  if (!normalizedValue || isLikelyHubId(value)) return null

  for (const root of roots) {
    if (!root.exists) continue
    const normalizedRoot = normalizeFsPath(root.path)
    if (normalizedValue === normalizedRoot) {
      return { rootId: root.id, browsePath: '.' }
    }
    if (normalizedValue.startsWith(`${normalizedRoot}/`)) {
      return {
        rootId: root.id,
        browsePath: normalizedValue.slice(normalizedRoot.length + 1),
      }
    }
  }

  return null
}

function isLikelyHubId(value: string): boolean {
  const trimmed = value.trim()
  if (!trimmed || trimmed.includes('\\')) return false
  if (trimmed.startsWith('/') || trimmed.startsWith('./') || trimmed.startsWith('../')) return false
  if (/^[A-Za-z]:[\\/]/.test(trimmed)) return false
  if (/^(output|data|uploads)\//i.test(trimmed)) return false

  const segments = trimmed.split('/')
  return segments.length === 2 && segments.every(Boolean)
}

function buildValidation(result: Record<string, unknown>, purpose: 'model' | 'adapter'): ModelPathValidation {
  const isAdapter = result.is_adapter === true
  if (purpose === 'adapter') {
    return {
      tone: isAdapter ? 'success' : 'warning',
      message: isAdapter
        ? 'Valid adapter folder on the backend.'
        : 'This folder does not look like a LoRA adapter. Use it directly as Model Path if it is a merged model.',
      isAdapter,
      isLocalPath: true,
    }
  }

  return {
    tone: isAdapter ? 'warning' : 'success',
    message: isAdapter
      ? 'This folder looks like a LoRA adapter. Put the base model in Model Path and this folder in Adapter Path.'
      : 'Valid model folder on the backend.',
    isAdapter,
    isLocalPath: true,
  }
}

function ValidationBadge({ validation }: { validation: ModelPathValidation }) {
  const Icon =
    validation.tone === 'success'
      ? CheckCircle2
      : validation.tone === 'info'
        ? Info
        : AlertCircle
  const variant =
    validation.tone === 'success'
      ? 'success'
      : validation.tone === 'warning'
        ? 'warning'
        : validation.tone === 'error'
          ? 'error'
          : 'outline'

  return (
    <div className="flex items-start gap-2 text-xs">
      <Badge variant={variant} className="gap-1 px-2 py-0.5">
        <Icon className="h-3 w-3" />
        {validation.tone === 'success'
          ? 'Validated'
          : validation.tone === 'warning'
            ? 'Check'
            : validation.tone === 'error'
              ? 'Invalid'
              : 'Info'}
      </Badge>
      <p className="pt-0.5 text-muted-foreground">{validation.message}</p>
    </div>
  )
}

export function ModelPathField({
  value,
  onChange,
  disabled = false,
  placeholder = 'meta-llama/Llama-3-8B',
  helperText,
  validationPurpose = 'model',
  onValidationChange,
}: ModelPathFieldProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [validation, setValidation] = useState<ModelPathValidation | null>(null)
  const [selectedRootId, setSelectedRootId] = useState('')
  const [browsePath, setBrowsePath] = useState('.')
  const [didAutoSyncOnOpen, setDidAutoSyncOnOpen] = useState(false)
  const { mutateAsync: executeTool, isPending: isValidating } = useToolExecution()
  const [rootMenuOpen, setRootMenuOpen] = useState(false)
  const { data: roots = [], isLoading: rootsLoading, isError: rootsError } = useDeploymentBrowseRoots()
  const {
    data: browseResult,
    isLoading: browseLoading,
    error: browseError,
  } = useDeploymentBrowseDir(selectedRootId, browsePath, open)

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  useEffect(() => {
    if (!open) {
      setDidAutoSyncOnOpen(false)
    }
  }, [open])

  useEffect(() => {
    setValidation(null)
    onValidationChange?.(null)
  }, [value, onValidationChange])

  useEffect(() => {
    if (!open || !roots.length) return
    if (didAutoSyncOnOpen) return

    const resolvedFromValue = resolveValueToBrowseLocation(value, roots)
    if (resolvedFromValue) {
      if (resolvedFromValue.rootId !== selectedRootId) setSelectedRootId(resolvedFromValue.rootId)
      if (resolvedFromValue.browsePath !== browsePath) setBrowsePath(resolvedFromValue.browsePath)
      setDidAutoSyncOnOpen(true)
      return
    }

    const nextRootId = preferredRootId(roots, validationPurpose)
    if (nextRootId && nextRootId !== selectedRootId) {
      setSelectedRootId(nextRootId)
      setBrowsePath('.')
    }
    setDidAutoSyncOnOpen(true)
  }, [open, roots, value, validationPurpose, selectedRootId, browsePath, didAutoSyncOnOpen])

  async function validatePath(nextValue?: string) {
    const target = (nextValue ?? value).trim()
    if (!target) {
      setValidation(null)
      onValidationChange?.(null)
      return
    }

    if (isLikelyHubId(target)) {
      const hubValidation: ModelPathValidation = {
        tone: 'info',
        message: 'This looks like a Hugging Face model ID. That is allowed, but local folder validation does not apply.',
        isAdapter: false,
        isLocalPath: false,
      }
      setValidation(hubValidation)
      onValidationChange?.(hubValidation)
      return
    }

    try {
      const result = await executeTool({
        toolName: 'validate.model_info',
        args: { model_path: target },
      }) as Record<string, unknown>
      const nextValidation = buildValidation(result, validationPurpose)
      setValidation(nextValidation)
      onValidationChange?.(nextValidation)
    } catch (error) {
      const nextValidation: ModelPathValidation = {
        tone: 'error',
        message: error instanceof Error ? error.message : 'Validation failed',
        isAdapter: false,
        isLocalPath: true,
      }
      setValidation(nextValidation)
      onValidationChange?.(nextValidation)
    }
  }

  async function handleSelect(candidatePath: string) {
    onChange(candidatePath)
    setOpen(false)
    await validatePath(candidatePath)
  }

  const currentAbsolutePath = browseResult?.current_absolute_path ?? ''
  const hasParent = typeof browseResult?.parent_path === 'string'

  return (
    <div ref={containerRef} className="space-y-2">
      <div className="flex gap-2">
        <Input
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onBlur={() => {
            if (value.trim()) void validatePath()
          }}
          placeholder={placeholder}
          disabled={disabled}
          className="flex-1"
        />
        <Button
          type="button"
          variant="outline"
          onClick={() => setOpen((current) => !current)}
          disabled={disabled}
          aria-label="Browse model folders"
        >
          <FolderSearch className="h-4 w-4" />
          Browse
          <ChevronDown className={cn('h-4 w-4 transition-transform', open && 'rotate-180')} />
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={() => void validatePath()}
          disabled={disabled || !value.trim() || isValidating}
        >
          {isValidating ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          Validate
        </Button>
      </div>

      {open && (
        <div className="rounded-md border border-border bg-card shadow-sm">
          {rootsLoading ? (
            <div className="flex items-center gap-2 px-3 py-4 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Connecting to gateway...
            </div>
          ) : rootsError ? (
            <div className="space-y-2 px-3 py-3">
              <p className="text-sm text-muted-foreground">
                Could not connect to the MCP gateway. Start it with:
              </p>
              <code className="block rounded bg-muted px-2 py-1 text-xs">python scripts/run_gateway.py</code>
              <p className="text-xs text-muted-foreground">
                Or type a path directly in the field above. For HF cached models, the default location is{' '}
                <code className="text-foreground">~/.cache/huggingface/hub</code>
              </p>
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setRootMenuOpen((c) => !c)}
                    className="flex h-8 items-center gap-1.5 rounded-md border border-input bg-background px-2 text-xs hover:bg-accent transition-colors"
                  >
                    {roots.find((r) => r.id === selectedRootId)?.label ?? 'Select root...'}
                    <ChevronDown className={cn('h-3 w-3 transition-transform', rootMenuOpen && 'rotate-180')} />
                  </button>
                  {rootMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-40" onClick={() => setRootMenuOpen(false)} />
                      <div className="absolute top-full mt-1 left-0 z-50 min-w-36 rounded-lg border bg-popover shadow-lg py-1">
                        {roots.map((root) => (
                          <button
                            key={root.id}
                            type="button"
                            disabled={!root.exists}
                            onClick={() => {
                              setSelectedRootId(root.id)
                              setBrowsePath('.')
                              setRootMenuOpen(false)
                            }}
                            className={cn(
                              'w-full text-left px-3 py-1.5 text-xs transition-colors',
                              root.exists ? 'hover:bg-accent text-foreground' : 'opacity-50 cursor-not-allowed text-muted-foreground',
                              root.id === selectedRootId && 'text-primary font-medium',
                            )}
                          >
                            {root.label}{root.exists ? '' : ' (missing)'}
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    if (browseResult?.parent_path) setBrowsePath(browseResult.parent_path)
                  }}
                  disabled={!hasParent}
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                  Up
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    if (currentAbsolutePath) void handleSelect(currentAbsolutePath)
                  }}
                  disabled={!currentAbsolutePath}
                >
                  Use Current Folder
                </Button>
              </div>
              <div className="border-b border-border px-3 py-2 text-xs text-muted-foreground" title={currentAbsolutePath || undefined}>
                {currentAbsolutePath
                  ? currentAbsolutePath
                  : selectedRootId
                    ? 'Loading folder contents...'
                    : 'Select a root to start browsing.'}
              </div>
              <div className="max-h-56 overflow-y-auto">
                {browseLoading ? (
                  <div className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Loading...
                  </div>
                ) : browseError ? (
                  <div className="px-3 py-2 text-sm text-red-400">
                    {browseError instanceof Error ? browseError.message : 'Browse failed'}
                  </div>
                ) : (browseResult?.entries ?? []).length === 0 ? (
                  <div className="px-3 py-2 text-sm text-muted-foreground">No folders in this location.</div>
                ) : (
                  (browseResult?.entries ?? []).map((entry) => (
                    <button
                      key={`${entry.type}:${entry.absolute_path}`}
                      type="button"
                      className="w-full cursor-pointer px-3 py-2 text-left transition-colors hover:bg-accent"
                      onClick={() => {
                        if (entry.type === 'directory') {
                          setBrowsePath(entry.path)
                        }
                      }}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <div className="text-sm font-medium">{entry.name}</div>
                          <div className="truncate font-mono text-[11px] text-muted-foreground" title={entry.absolute_path}>
                            {entry.absolute_path}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={entry.type === 'directory' ? 'outline' : 'secondary'}>
                            {entry.type}
                          </Badge>
                          {entry.type === 'directory' && (
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={(event) => {
                                event.stopPropagation()
                                void handleSelect(entry.absolute_path)
                              }}
                            >
                              Select
                            </Button>
                          )}
                        </div>
                      </div>
                    </button>
                  ))
                )}
              </div>
              <div className="border-t border-border px-3 py-2 text-xs text-muted-foreground">
                Browse navigates server-visible folders. Select the target directory with <code>Use Current Folder</code>.
              </div>
            </>
          )}
        </div>
      )}

      {validation ? <ValidationBadge validation={validation} /> : null}
      {helperText ? <p className="text-xs text-muted-foreground">{helperText}</p> : null}
    </div>
  )
}
