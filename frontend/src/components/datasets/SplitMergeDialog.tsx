import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { toast } from 'sonner'

interface SplitMergeDialogProps {
  open: boolean
  onClose: () => void
  mode: 'split' | 'merge'
  datasetPath?: string
  datasetPaths?: string[]
}

export function SplitMergeDialog({
  open,
  onClose,
  mode,
  datasetPath,
  datasetPaths,
}: SplitMergeDialogProps) {
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()

  const [splitRatio, setSplitRatio] = useState(0.8)
  const [outputDir, setOutputDir] = useState('data')
  const [outputName, setOutputName] = useState('')

  useEffect(() => {
    if (mode !== 'merge' || !datasetPaths?.length || outputName.trim()) {
      return
    }

    const baseNames = datasetPaths.map((p) => p.split(/[\\/]/).pop()?.replace(/\.jsonl?$/i, '') ?? 'dataset')
    const sharedPrefix = baseNames.reduce((prefix, name) => {
      let i = 0
      while (i < prefix.length && i < name.length && prefix[i] === name[i]) {
        i += 1
      }
      return prefix.slice(0, i).replace(/[_-]+$/g, '')
    })

    const suggested = sharedPrefix
      ? `data/${sharedPrefix}_merged`
      : `data/merged_${datasetPaths.length}_datasets`

    setOutputName(suggested)
  }, [mode, datasetPaths, outputName])

  async function handleExecute() {
    try {
      if (mode === 'split' && datasetPath) {
        const valRatio = Number(((1 - splitRatio) / 2).toFixed(2))
        const testRatio = Number((1 - splitRatio - valRatio).toFixed(2))
        await executeTool({
          toolName: 'dataset.split',
          args: {
            file_path: datasetPath,
            output_dir: outputDir,
            train_ratio: splitRatio,
            val_ratio: valRatio,
            test_ratio: testRatio,
          },
        })
        toast.success('Dataset split successfully')
      } else if (mode === 'merge' && datasetPaths?.length) {
        const safeName = outputName.trim() || 'merged_dataset'
        await executeTool({
          toolName: 'dataset.merge',
          args: {
            file_paths: datasetPaths,
            output_path: safeName.endsWith('.jsonl') ? safeName : `${safeName}.jsonl`,
          },
        })
        toast.success('Datasets merged successfully')
      }
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      onClose()
    } catch (err) {
      toast.error(`${mode} failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  const title = mode === 'split' ? 'Split Dataset' : 'Merge Datasets'

  return (
    <Dialog open={open} onClose={onClose} title={title}>
      <div className="space-y-4">
        {mode === 'split' && datasetPath && (
          <>
            <div>
              <p className="text-sm text-muted-foreground mb-2">
                Splitting: <strong>{datasetPath.split(/[\\/]/).pop()}</strong>
              </p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Split ratio</label>
              <div className="flex items-center gap-3">
                <Input
                  type="number"
                  value={splitRatio}
                  onChange={(e) => setSplitRatio(Number(e.target.value))}
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  className="w-24"
                />
                <div className="flex gap-2">
                  <Badge variant="secondary">Train: {Math.round(splitRatio * 100)}%</Badge>
                  <Badge variant="outline">Test: {Math.round((1 - splitRatio) * 100)}%</Badge>
                </div>
              </div>
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Output directory</label>
              <Input
                placeholder="data/splits"
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
              />
            </div>
          </>
        )}

        {mode === 'merge' && datasetPaths && (
          <>
            <div className="space-y-2">
              <p className="text-sm font-medium">Datasets to merge</p>
              <p className="text-xs text-muted-foreground">
                {datasetPaths.length} dataset{datasetPaths.length === 1 ? '' : 's'} selected
              </p>
              <div className="flex flex-wrap gap-1">
                {datasetPaths.map((p) => (
                  <Badge key={p} variant="secondary">
                    {p.split(/[\\/]/).pop()}
                  </Badge>
                ))}
              </div>
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Output path</label>
              <Input
                placeholder="data/merged_dataset"
                value={outputName}
                onChange={(e) => setOutputName(e.target.value)}
              />
            </div>
          </>
        )}

        <div className="flex justify-end gap-2 pt-2">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleExecute} disabled={isPending}>
            {mode === 'split' ? 'Split' : 'Merge'}
          </Button>
        </div>
      </div>
    </Dialog>
  )
}
