import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { DatasetInfo } from '@/api/types'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { formatBytes } from '@/lib/utils'
import { Eye, EyeOff, Download, Trash2, Scissors } from 'lucide-react'
import { toast } from 'sonner'
import { Dialog } from '@/components/ui/dialog'
import { DatasetPreview } from './DatasetPreview'
import { SplitMergeDialog } from './SplitMergeDialog'

interface DatasetCardProps {
  dataset: DatasetInfo
}

export function DatasetCard({ dataset }: DatasetCardProps) {
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const [showPreview, setShowPreview] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [showSplit, setShowSplit] = useState(false)

  const fileName = dataset.file_path.split(/[\\/]/).pop() ?? dataset.file_path

  async function handleExport() {
    try {
      await executeTool({ toolName: 'dataset.export', args: { file_path: dataset.file_path } })
      toast.success(`Exported ${fileName}`)
    } catch (err) {
      toast.error(`Export failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleDelete() {
    try {
      await executeTool({ toolName: 'dataset.delete', args: { file_path: dataset.file_path } })
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success(`Deleted ${fileName}`)
      setShowDeleteConfirm(false)
    } catch (err) {
      toast.error(`Delete failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  return (
    <>
      <Card>
        <CardContent className="p-4 space-y-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="font-semibold text-sm truncate" title={dataset.file_path}>
              {fileName}
            </h3>
            <Badge variant="secondary">{dataset.format}</Badge>
          </div>

          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <span>{dataset.row_count.toLocaleString()} rows</span>
            <span>{formatBytes(dataset.size_bytes)}</span>
          </div>

          {dataset.technique && (
            <Badge variant="outline" className="text-xs">
              {dataset.technique}
            </Badge>
          )}

          {dataset.columns.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {dataset.columns.slice(0, 3).map((col) => (
                <Badge key={col} variant="secondary" className="text-[10px] px-1.5 py-0">
                  {col}
                </Badge>
              ))}
              {dataset.columns.length > 3 && (
                <span className="text-[10px] text-muted-foreground">
                  +{dataset.columns.length - 3} more
                </span>
              )}
            </div>
          )}

          <div className="flex items-center gap-1 pt-1 border-t border-border">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowPreview(!showPreview)}
              disabled={isPending}
            >
              {showPreview ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
              View
            </Button>
            <Button variant="ghost" size="sm" onClick={handleExport} disabled={isPending}>
              <Download className="h-3.5 w-3.5" />
              Export
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setShowSplit(true)} disabled={isPending}>
              <Scissors className="h-3.5 w-3.5" />
              Split
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="ml-auto text-destructive hover:text-destructive"
              onClick={() => setShowDeleteConfirm(true)}
              disabled={isPending}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>

          {showPreview && <DatasetPreview filePath={dataset.file_path} />}
        </CardContent>
      </Card>

      <Dialog
        open={showDeleteConfirm}
        onClose={() => setShowDeleteConfirm(false)}
        title="Delete Dataset"
      >
        <p className="text-sm text-muted-foreground mb-4">
          Are you sure you want to delete <strong>{fileName}</strong>? This action cannot be undone.
        </p>
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => setShowDeleteConfirm(false)}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={handleDelete} disabled={isPending}>
            Delete
          </Button>
        </div>
      </Dialog>

      <SplitMergeDialog
        open={showSplit}
        onClose={() => setShowSplit(false)}
        mode="split"
        datasetPath={dataset.file_path}
      />
    </>
  )
}
