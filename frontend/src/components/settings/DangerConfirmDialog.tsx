import { useState, useEffect } from 'react'
import { Dialog } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { AlertTriangle } from 'lucide-react'

interface DangerConfirmDialogProps {
  open: boolean
  onClose: () => void
  onConfirm: () => void
  title: string
  description: string
  confirmText: string
}

export function DangerConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  description,
  confirmText,
}: DangerConfirmDialogProps) {
  const [inputValue, setInputValue] = useState('')

  useEffect(() => {
    if (!open) setInputValue('')
  }, [open])

  const matches = inputValue === confirmText

  return (
    <Dialog open={open} onClose={onClose} title={title}>
      <div className="space-y-4">
        <div className="flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-3">
          <AlertTriangle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>

        <div className="space-y-2">
          <label className="text-sm text-muted-foreground">
            Type <span className="font-mono font-semibold text-foreground">{confirmText}</span> to
            confirm
          </label>
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={confirmText}
            autoFocus
          />
        </div>

        <div className="flex justify-end gap-2 pt-2">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="destructive" disabled={!matches} onClick={onConfirm}>
            Confirm
          </Button>
        </div>
      </div>
    </Dialog>
  )
}
