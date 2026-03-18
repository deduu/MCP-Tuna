import { useState, useRef, useEffect } from 'react'
import { ChevronDown, Search, Star, Download, Sparkles } from 'lucide-react'
import { useHFSearch, useLocalModelCandidates, useRecommendedModels } from '@/api/hooks/useTraining'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { inferModelModality } from '@/lib/training-capabilities'

type BrowseMode = 'local' | 'hub' | 'recommended'

interface ModelBrowserProps {
  value: string
  onChange: (model: string) => void
}

export function ModelBrowser({ value, onChange }: ModelBrowserProps) {
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<BrowseMode>('local')
  const [hubQuery, setHubQuery] = useState('')
  const [hubSearchEnabled, setHubSearchEnabled] = useState(false)
  const [useCase, setUseCase] = useState('general')
  const containerRef = useRef<HTMLDivElement>(null)

  const { data: models = [], isLoading: localLoading } = useLocalModelCandidates(value)
  const { data: hubResults, isLoading: hubLoading } = useHFSearch(hubQuery, 'text-generation', hubSearchEnabled)
  const { data: recResults, isLoading: recLoading } = useRecommendedModels(useCase)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleHubSearch = () => {
    if (hubQuery.length >= 2) setHubSearchEnabled(true)
  }

  const selectModel = (model: string) => {
    onChange(model)
    setOpen(false)
  }

  return (
    <div ref={containerRef} className="relative w-full">
      <div className="flex gap-1">
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Model path or HuggingFace ID..."
          className="flex-1"
          onFocus={() => setOpen(true)}
        />
        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={() => setOpen((o) => !o)}
          aria-label="Browse models"
        >
          <ChevronDown className={cn('h-4 w-4 transition-transform', open && 'rotate-180')} />
        </Button>
      </div>

      {open && (
        <div className="absolute z-50 mt-1 w-full rounded-md border border-border bg-card shadow-lg max-h-72 overflow-hidden flex flex-col">
          {/* Tab buttons */}
          <div className="flex border-b border-border shrink-0">
            {([
              { key: 'local', label: 'Local', icon: Download },
              { key: 'hub', label: 'HuggingFace', icon: Search },
              { key: 'recommended', label: 'Recommended', icon: Sparkles },
            ] as const).map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                type="button"
                className={cn(
                  'flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs transition-colors',
                  mode === key
                    ? 'bg-accent text-accent-foreground font-medium'
                    : 'text-muted-foreground hover:text-foreground',
                )}
                onClick={() => setMode(key)}
              >
                <Icon className="h-3 w-3" />
                {label}
              </button>
            ))}
          </div>

          <div className="overflow-y-auto flex-1">
            {/* Local tab */}
            {mode === 'local' && (
              localLoading ? (
                <div className="px-3 py-2 text-sm text-muted-foreground">Loading models...</div>
              ) : models.length === 0 ? (
                <div className="px-3 py-2 text-sm text-muted-foreground">No local models found</div>
              ) : (
                models.map((model) => {
                  const label = model.model_path ?? model.id
                  const modality = inferModelModality(label, model)

                  return (
                    <button
                      key={model.id}
                      type="button"
                      className="w-full cursor-pointer px-3 py-2 text-left text-sm hover:bg-accent transition-colors"
                      onClick={() => selectModel(label)}
                    >
                      <div className="font-medium">{label}</div>
                      <div className="mt-0.5 text-xs text-muted-foreground">
                        {modality === 'vision-language' ? 'Vision-language model' : 'Text model'}
                      </div>
                    </button>
                  )
                })
              )
            )}

            {/* HuggingFace Hub tab */}
            {mode === 'hub' && (
              <>
                <div className="flex gap-1 p-2 border-b border-border">
                  <Input
                    value={hubQuery}
                    onChange={(e) => {
                      setHubQuery(e.target.value)
                      setHubSearchEnabled(false)
                    }}
                    placeholder="Search HuggingFace..."
                    className="flex-1 h-7 text-xs"
                    onKeyDown={(e) => e.key === 'Enter' && handleHubSearch()}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 px-2"
                    onClick={handleHubSearch}
                    disabled={hubLoading || hubQuery.length < 2}
                  >
                    <Search className="h-3 w-3" />
                  </Button>
                </div>
                {hubLoading ? (
                  <div className="px-3 py-2 text-sm text-muted-foreground">Searching...</div>
                ) : !hubResults?.models?.length ? (
                  <div className="px-3 py-2 text-sm text-muted-foreground">
                    {hubSearchEnabled ? 'No models found' : 'Type a query and press Enter'}
                  </div>
                ) : (
                  hubResults.models.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      className="w-full cursor-pointer px-3 py-2 text-left hover:bg-accent transition-colors"
                      onClick={() => selectModel(m.id)}
                    >
                      <div className="text-sm font-medium">{m.id}</div>
                      <div className="flex gap-3 text-xs text-muted-foreground mt-0.5">
                        <span className="flex items-center gap-0.5">
                          <Download className="h-3 w-3" />
                          {m.downloads?.toLocaleString()}
                        </span>
                        <span className="flex items-center gap-0.5">
                          <Star className="h-3 w-3" />
                          {m.likes}
                        </span>
                        {m.library && <span>{m.library}</span>}
                      </div>
                    </button>
                  ))
                )}
              </>
            )}

            {/* Recommended tab */}
            {mode === 'recommended' && (
              <>
                <div className="p-2 border-b border-border">
                  <select
                    value={useCase}
                    onChange={(e) => setUseCase(e.target.value)}
                    className="w-full h-7 rounded-md border border-border bg-background px-2 text-xs"
                  >
                    <option value="general">General</option>
                    <option value="low_memory">Low Memory</option>
                    <option value="speed">Speed</option>
                    <option value="quality">Quality</option>
                    <option value="multilingual">Multilingual</option>
                    <option value="indonesian">Indonesian</option>
                  </select>
                </div>
                {recLoading ? (
                  <div className="px-3 py-2 text-sm text-muted-foreground">Loading...</div>
                ) : !recResults?.recommendations?.length ? (
                  <div className="px-3 py-2 text-sm text-muted-foreground">No recommendations</div>
                ) : (
                  recResults.recommendations.map((r) => (
                    <button
                      key={r.model_id}
                      type="button"
                      className="w-full cursor-pointer px-3 py-2 text-left hover:bg-accent transition-colors"
                      onClick={() => selectModel(r.model_id)}
                    >
                      <div className="text-sm font-medium">{r.model_id}</div>
                      <div className="flex gap-3 text-xs text-muted-foreground mt-0.5">
                        <span>{r.size}</span>
                        <span>{r.memory}</span>
                        <span>{r.description}</span>
                      </div>
                    </button>
                  ))
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
