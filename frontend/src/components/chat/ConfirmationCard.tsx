import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { ShieldAlert, ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { ConfirmationRequest } from '@/stores/chat'

interface ConfirmationCardProps {
  confirmation: ConfirmationRequest
  onProceed: () => void
  onCancel: () => void
  onModify: () => void
  disabled?: boolean
}

export function ConfirmationCard({
  confirmation,
  onProceed,
  onCancel,
  onModify,
  disabled,
}: ConfirmationCardProps) {
  const [showArgs, setShowArgs] = useState(false)
  const argEntries = Object.entries(confirmation.arguments ?? {})

  return (
    <div className="my-2 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 space-y-3">
      <div className="flex items-start gap-2">
        <ShieldAlert className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
        <div className="space-y-1 min-w-0">
          <p className="text-sm font-medium">{confirmation.message}</p>
          <p className="text-xs text-muted-foreground">
            Tool: <code className="bg-muted px-1 rounded">{confirmation.tool}</code>
          </p>
        </div>
      </div>

      {argEntries.length > 0 && (
        <div>
          <button
            type="button"
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            onClick={() => setShowArgs((s) => !s)}
          >
            <ChevronDown className={cn('h-3 w-3 transition-transform', showArgs && 'rotate-180')} />
            Arguments
          </button>
          {showArgs && (
            <pre className="mt-1 max-h-32 overflow-auto rounded bg-muted p-2 text-xs">
              {JSON.stringify(confirmation.arguments, null, 2)}
            </pre>
          )}
        </div>
      )}

      <div className="flex gap-2">
        <Button
          size="sm"
          onClick={onProceed}
          disabled={disabled}
          className="bg-green-600 hover:bg-green-700 text-white"
        >
          Proceed
        </Button>
        <Button size="sm" variant="destructive" onClick={onCancel} disabled={disabled}>
          Cancel
        </Button>
        <Button size="sm" variant="outline" onClick={onModify} disabled={disabled}>
          Modify
        </Button>
      </div>
    </div>
  )
}
