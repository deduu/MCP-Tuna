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
  selected?: boolean
  onToggleSelect?: (filePath: string, selected: boolean) => void
}

export function DatasetCard({ dataset, selected = false, onToggleSelect }: DatasetCardProps) {
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const [showPreview, setShowPreview] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [showSplit, setShowSplit] = useState(false)

  const fileName = dataset.file_path.split(/[\\/]/).pop() ?? dataset.file_path
  const fileBaseName = fileName.includes('.') ? fileName.slice(0, fileName.lastIndexOf('.')) : fileName

  function downloadText(content: string, downloadName: string, mimeType: string) {
    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = downloadName
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  async function handleExport() {
    try {
      const loaded = await executeTool({
        toolName: 'dataset.load',
        args: { file_path: dataset.file_path },
      })
      const payload = loaded as Record<string, unknown>
      const points = Array.isArray(payload.data_points)
        ? (payload.data_points as Array<Record<string, unknown>>)
        : []
      if (points.length === 0) {
        throw new Error('Dataset has no rows to export')
      }

      const fmt = dataset.format.toLowerCase()
      if (fmt === 'json') {
        const outName = fileName.endsWith('.json') ? fileName : `${fileBaseName}.json`
        downloadText(JSON.stringify(points, null, 2), outName, 'application/json;charset=utf-8')
        toast.success(`Exported ${outName}`)
        return
      }

      const outName = fileName.endsWith('.jsonl') ? fileName : `${fileBaseName}.jsonl`
      const content = points.map((row) => JSON.stringify(row)).join('\n')
      downloadText(content, outName, 'application/x-ndjson;charset=utf-8')
      if (fmt !== 'jsonl') {
        toast.success(`Exported ${outName} (converted from ${dataset.format})`)
      } else {
        toast.success(`Exported ${outName}`)
      }
    } catch (err) {
      toast.error(`Export failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleDelete() {
    try {
      await executeTool({
        toolName: 'dataset.delete',
        args: { file_path: dataset.file_path },
      })
      await queryClient.invalidateQueries({ queryKey: ['datasets'] })
      setShowDeleteConfirm(false)
      toast.success(`Deleted ${fileName}`)
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
            <div className="flex items-center gap-2 shrink-0">
              {onToggleSelect && (
                <label className="flex items-center gap-1 text-[11px] text-muted-foreground cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selected}
                    onChange={(e) => onToggleSelect(dataset.file_path, e.target.checked)}
                    className="rounded border-input"
                  />
                  Merge
                </label>
              )}
              <Badge variant="secondary">{dataset.format}</Badge>
            </div>
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
              title={`Delete ${fileName}`}
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
