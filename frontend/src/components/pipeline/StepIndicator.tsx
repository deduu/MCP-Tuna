import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'

interface StepIndicatorProps {
  steps: string[]
  currentStep: number
  className?: string
}

export function StepIndicator({ steps, currentStep, className }: StepIndicatorProps) {
  return (
    <div className={cn('flex flex-wrap items-start gap-y-4', className)}>
      {steps.map((label, i) => {
        const isCompleted = i < currentStep
        const isCurrent = i === currentStep
        const isFuture = i > currentStep

        return (
          <div key={label} className="flex items-center">
            <div className="flex flex-col items-center gap-1.5">
              <div
                className={cn(
                  'flex h-8 w-8 items-center justify-center rounded-full text-xs font-semibold transition-colors',
                  isCompleted && 'bg-emerald-500/20 text-emerald-400',
                  isCurrent && 'bg-primary text-primary-foreground animate-pulse',
                  isFuture && 'bg-secondary text-secondary-foreground',
                )}
              >
                {isCompleted ? <Check className="h-4 w-4" /> : i + 1}
              </div>
              <span
                className={cn(
                  'text-[11px] max-w-[5rem] text-center leading-tight',
                  isCurrent ? 'text-foreground font-medium' : 'text-muted-foreground',
                )}
              >
                {label}
              </span>
            </div>

            {i < steps.length - 1 && (
              <div
                className={cn(
                  'mx-1 mt-[-1rem] h-0.5 w-6 sm:w-10',
                  i < currentStep ? 'bg-emerald-500/40' : 'bg-secondary',
                )}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}
