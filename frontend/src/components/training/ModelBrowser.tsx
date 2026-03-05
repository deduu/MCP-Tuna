import { useState, useRef, useEffect } from 'react'
import { ChevronDown } from 'lucide-react'
import { useAvailableModels } from '@/api/hooks/useTraining'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface ModelBrowserProps {
  value: string
  onChange: (model: string) => void
}

export function ModelBrowser({ value, onChange }: ModelBrowserProps) {
  const [open, setOpen] = useState(false)
  const { data: models = [], isLoading } = useAvailableModels()
  const containerRef = useRef<HTMLDivElement>(null)

  const filtered = models.filter((m) =>
    m.toLowerCase().includes(value.toLowerCase()),
  )

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <div ref={containerRef} className="relative w-full">
      <div className="flex gap-1">
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Model path..."
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
        <div className="absolute z-50 mt-1 w-full rounded-md border border-border bg-card shadow-lg max-h-48 overflow-y-auto">
          {isLoading ? (
            <div className="px-3 py-2 text-sm text-muted-foreground">Loading models...</div>
          ) : filtered.length === 0 ? (
            <div className="px-3 py-2 text-sm text-muted-foreground">No models found</div>
          ) : (
            filtered.map((model) => (
              <button
                key={model}
                type="button"
                className="w-full cursor-pointer px-3 py-2 text-left text-sm hover:bg-accent transition-colors"
                onClick={() => {
                  onChange(model)
                  setOpen(false)
                }}
              >
                {model}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}
