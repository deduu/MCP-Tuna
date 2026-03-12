import { useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronLeft, FolderSearch, Loader2 } from 'lucide-react'
import { useDeploymentBrowseDir, useDeploymentBrowseRoots } from '@/api/hooks/useTraining'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

type DirectorySelectionMode = 'replace' | 'append-filename'

interface BrowsePathFieldProps {
  value: string
  onChange: (value: string) => void
  disabled?: boolean
  placeholder?: string
  helperText?: string
  allowFiles?: boolean
  allowDirectories?: boolean
  preferredRootIds?: string[]
  directorySelectionMode?: DirectorySelectionMode
  defaultFileName?: string
  browseLabel?: string
}

function normalizeFsPath(value: string): string {
  return value.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase()
}

function parentBrowsePath(path: string): string {
  const segments = path.split('/').filter(Boolean)
  if (segments.length <= 1) return '.'
  return segments.slice(0, -1).join('/')
}

function joinPath(directoryPath: string, filename: string): string {
  if (!filename) return directoryPath
  const separator = directoryPath.includes('\\') ? '\\' : '/'
  return `${directoryPath.replace(/[\\/]+$/, '')}${separator}${filename}`
}

function extractFileName(value: string): string {
  const trimmed = value.trim().replace(/[\\/]+$/, '')
  if (!trimmed) return ''
  const segments = trimmed.split(/[\\/]/).filter(Boolean)
  return segments.at(-1) ?? ''
}

function preferredRootId(
  roots: Array<{ id: string; exists: boolean }>,
  preferredRootIds: string[],
): string {
  for (const rootId of preferredRootIds) {
    const match = roots.find((root) => root.id === rootId && root.exists)
    if (match) return match.id
  }

  return roots.find((root) => root.exists)?.id ?? roots[0]?.id ?? ''
}

function resolveValueToBrowseLocation(
  value: string,
  roots: Array<{ id: string; path: string; exists: boolean }>,
): { rootId: string; browsePath: string } | null {
  const trimmed = value.trim()
  const normalizedValue = normalizeFsPath(trimmed)
  if (!normalizedValue) return null

  for (const root of roots) {
    if (!root.exists) continue
    const normalizedRoot = normalizeFsPath(root.path)
    if (normalizedValue === normalizedRoot) {
      return { rootId: root.id, browsePath: '.' }
    }
    if (normalizedValue.startsWith(`${normalizedRoot}/`)) {
      return {
        rootId: root.id,
        browsePath: parentBrowsePath(normalizedValue.slice(normalizedRoot.length + 1)),
      }
    }
  }

  if (/^[A-Za-z]:[\\/]/.test(trimmed) || trimmed.startsWith('/')) {
    return null
  }

  const normalizedRelative = normalizeFsPath(trimmed)
  const relativeSegments = normalizedRelative.split('/').filter(Boolean)
  if (relativeSegments.length === 0) return null

  const firstSegment = relativeSegments[0]
  const matchingRoot = roots.find((root) => root.exists && root.id === firstSegment)
  if (matchingRoot) {
    return {
      rootId: matchingRoot.id,
      browsePath: parentBrowsePath(relativeSegments.slice(1).join('/')),
    }
  }

  const workspaceRoot = roots.find((root) => root.exists && root.id === 'workspace')
  if (!workspaceRoot) return null

  return {
    rootId: workspaceRoot.id,
    browsePath: parentBrowsePath(normalizedRelative),
  }
}

export function BrowsePathField({
  value,
  onChange,
  disabled = false,
  placeholder,
  helperText,
  allowFiles = true,
  allowDirectories = true,
  preferredRootIds = ['workspace', 'output', 'uploads', 'hf_cache'],
  directorySelectionMode = 'replace',
  defaultFileName,
  browseLabel = 'Browse',
}: BrowsePathFieldProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [selectedRootId, setSelectedRootId] = useState('')
  const [browsePath, setBrowsePath] = useState('.')
  const [locationCustomized, setLocationCustomized] = useState(false)
  const [rootMenuOpen, setRootMenuOpen] = useState(false)
  const { data: roots = [], isLoading: rootsLoading, isError: rootsError } = useDeploymentBrowseRoots()
  const autoLocation = resolveValueToBrowseLocation(value, roots)
  const defaultRootId = preferredRootId(roots, preferredRootIds)
  const activeRootId =
    open && !locationCustomized
      ? (autoLocation?.rootId ?? defaultRootId)
      : (selectedRootId || autoLocation?.rootId || defaultRootId)
  const activeBrowsePath =
    open && !locationCustomized
      ? (autoLocation?.browsePath ?? '.')
      : (browsePath || autoLocation?.browsePath || '.')
  const {
    data: browseResult,
    isLoading: browseLoading,
    error: browseError,
  } = useDeploymentBrowseDir(activeRootId, activeBrowsePath, open)

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false)
        setLocationCustomized(false)
        setRootMenuOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  function selectDirectory(directoryPath: string) {
    if (directorySelectionMode === 'append-filename') {
      const filename = extractFileName(value) || defaultFileName || ''
      onChange(filename ? joinPath(directoryPath, filename) : directoryPath)
      return
    }

    onChange(directoryPath)
  }

  function handleSelectPath(candidatePath: string, type: 'directory' | 'file') {
    if (type === 'directory') {
      selectDirectory(candidatePath)
    } else {
      onChange(candidatePath)
    }
    setOpen(false)
    setLocationCustomized(false)
    setRootMenuOpen(false)
  }

  const currentAbsolutePath = browseResult?.current_absolute_path ?? ''
  const hasParent = typeof browseResult?.parent_path === 'string'

  return (
    <div ref={containerRef} className="space-y-2">
      <div className="flex gap-2">
        <Input
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className="flex-1"
        />
        <Button
          type="button"
          variant="outline"
          onClick={() =>
            setOpen((current) => {
              const next = !current
              if (!next) {
                setLocationCustomized(false)
                setRootMenuOpen(false)
              }
              return next
            })
          }
          disabled={disabled}
        >
          <FolderSearch className="h-4 w-4" />
          {browseLabel}
          <ChevronDown className={cn('h-4 w-4 transition-transform', open && 'rotate-180')} />
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
                Or type a path directly in the field above.
              </p>
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setRootMenuOpen((current) => !current)}
                    className="flex h-8 items-center gap-1.5 rounded-md border border-input bg-background px-2 text-xs transition-colors hover:bg-accent"
                  >
                    {roots.find((root) => root.id === activeRootId)?.label ?? 'Select root...'}
                    <ChevronDown className={cn('h-3 w-3 transition-transform', rootMenuOpen && 'rotate-180')} />
                  </button>
                  {rootMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-40" onClick={() => setRootMenuOpen(false)} />
                      <div className="absolute left-0 top-full z-50 mt-1 min-w-36 rounded-lg border bg-popover py-1 shadow-lg">
                        {roots.map((root) => (
                          <button
                            key={root.id}
                            type="button"
                            disabled={!root.exists}
                    onClick={() => {
                      setLocationCustomized(true)
                      setSelectedRootId(root.id)
                      setBrowsePath('.')
                      setRootMenuOpen(false)
                            }}
                            className={cn(
                              'w-full px-3 py-1.5 text-left text-xs transition-colors',
                              root.exists
                                ? 'text-foreground hover:bg-accent'
                                : 'cursor-not-allowed text-muted-foreground opacity-50',
                              root.id === activeRootId && 'font-medium text-primary',
                            )}
                          >
                            {root.label}
                            {root.exists ? '' : ' (missing)'}
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
                    if (browseResult?.parent_path) {
                      setLocationCustomized(true)
                      setBrowsePath(browseResult.parent_path)
                    }
                  }}
                  disabled={!hasParent}
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                  Up
                </Button>
                {allowDirectories && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (currentAbsolutePath) handleSelectPath(currentAbsolutePath, 'directory')
                    }}
                    disabled={!currentAbsolutePath}
                  >
                    Use Current Folder
                  </Button>
                )}
              </div>
              <div className="border-b border-border px-3 py-2 text-xs text-muted-foreground">
                {currentAbsolutePath
                  ? currentAbsolutePath
                  : activeRootId
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
                  <div className="px-3 py-2 text-sm text-muted-foreground">No entries in this location.</div>
                ) : (
                  (browseResult?.entries ?? []).map((entry) => {
                    const canSelect =
                      (entry.type === 'directory' && allowDirectories) ||
                      (entry.type === 'file' && allowFiles)

                    return (
                      <button
                        key={`${entry.type}:${entry.absolute_path}`}
                        type="button"
                        className="w-full cursor-pointer px-3 py-2 text-left transition-colors hover:bg-accent"
                        onClick={() => {
                          if (entry.type === 'directory') {
                            setLocationCustomized(true)
                            setBrowsePath(entry.path)
                          }
                        }}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-sm font-medium">{entry.name}</div>
                            <div className="truncate font-mono text-[11px] text-muted-foreground">
                              {entry.absolute_path}
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant={entry.type === 'directory' ? 'outline' : 'secondary'}>
                              {entry.type}
                            </Badge>
                            {canSelect && (
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={(event) => {
                                  event.stopPropagation()
                                  handleSelectPath(entry.absolute_path, entry.type)
                                }}
                              >
                                Select
                              </Button>
                            )}
                          </div>
                        </div>
                      </button>
                    )
                  })
                )}
              </div>
              <div className="border-t border-border px-3 py-2 text-xs text-muted-foreground">
                Browse navigates server-visible folders and files exposed by the gateway.
              </div>
            </>
          )}
        </div>
      )}

      {helperText ? <p className="text-xs text-muted-foreground">{helperText}</p> : null}
    </div>
  )
}
