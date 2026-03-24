import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { DangerConfirmDialog } from './DangerConfirmDialog'
import { Trash2, ShieldAlert } from 'lucide-react'
import { toast } from 'sonner'

export function MaintenanceSection() {
  const [dialogOpen, setDialogOpen] = useState(false)
  const clearAll = useToolExecution()

  const handleConfirm = () => {
    setDialogOpen(false)
    clearAll.mutate(
      { toolName: 'system.clear_all', args: {} },
      {
        onSuccess: () => toast.success('Deployments stopped and GPU memory cleared.'),
        onError: (err) => toast.error(`Failed to reset runtime state: ${err.message}`),
      },
    )
  }

  return (
    <>
      <Card className="border-destructive/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <ShieldAlert className="h-4 w-4" />
            Maintenance
          </CardTitle>
          <CardDescription>
            Runtime recovery actions for stuck deployments or GPU memory issues.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between rounded-lg border border-destructive/20 bg-destructive/5 p-4">
            <div className="space-y-1">
              <p className="text-sm font-medium">Stop Deployments and Clear GPU</p>
              <p className="text-xs text-muted-foreground">
                Stops active deployments and clears GPU memory. Stored datasets and job history are not deleted.
              </p>
            </div>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => setDialogOpen(true)}
              disabled={clearAll.isPending}
            >
              <Trash2 className="h-3.5 w-3.5" />
              {clearAll.isPending ? 'Resetting...' : 'Reset Runtime'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <DangerConfirmDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onConfirm={handleConfirm}
        title="Reset Runtime State"
        description="This will stop active deployments and clear GPU memory for recovery. It does not delete datasets, training jobs, or cached files."
        confirmText="reset runtime"
      />
    </>
  )
}
