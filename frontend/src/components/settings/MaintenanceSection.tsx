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
        onSuccess: () => toast.success('All data has been cleared successfully.'),
        onError: (err) => toast.error(`Failed to clear data: ${err.message}`),
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
            Destructive actions that cannot be undone. Proceed with caution.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between rounded-lg border border-destructive/20 bg-destructive/5 p-4">
            <div className="space-y-1">
              <p className="text-sm font-medium">Clear All Data</p>
              <p className="text-xs text-muted-foreground">
                Remove all datasets, training jobs, deployments, and cached files.
              </p>
            </div>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => setDialogOpen(true)}
              disabled={clearAll.isPending}
            >
              <Trash2 className="h-3.5 w-3.5" />
              {clearAll.isPending ? 'Clearing...' : 'Clear All'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <DangerConfirmDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onConfirm={handleConfirm}
        title="Clear All Data"
        description="This will permanently delete all datasets, training jobs, model deployments, and cached files. This action cannot be undone."
        confirmText="delete everything"
      />
    </>
  )
}
