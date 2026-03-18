import type { TrainingTechnique } from '@/api/types'
import type { TrainingTechniqueOption } from '@/lib/training-capabilities'
import { cn } from '@/lib/utils'

interface TrainingTechniqueSelectorProps {
  options: TrainingTechniqueOption[]
  value: TrainingTechnique
  onChange: (value: TrainingTechnique) => void
}

export function TrainingTechniqueSelector({
  options,
  value,
  onChange,
}: TrainingTechniqueSelectorProps) {
  const selected = options.find((option) => option.value === value) ?? options[0]
  const disabledOptions = options.filter((option) => !option.enabled && option.reason)

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">Technique</label>
      <div className="flex rounded-lg border border-border overflow-hidden">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            disabled={!option.enabled}
            className={cn(
              'flex-1 px-3 py-2 text-sm font-medium transition-colors',
              option.value === value
                ? 'bg-primary text-primary-foreground'
                : 'bg-transparent text-muted-foreground hover:bg-accent',
              !option.enabled && 'cursor-not-allowed opacity-60 hover:bg-transparent',
              option.enabled && 'cursor-pointer',
            )}
          >
            {option.label}
          </button>
        ))}
      </div>
      {selected && (
        <p className="text-xs text-muted-foreground">
          {selected.reason ?? selected.description}
        </p>
      )}
      {disabledOptions.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Unavailable now: {disabledOptions.map((option) => `${option.label} (${option.reason})`).join(' ')}
        </p>
      )}
    </div>
  )
}
