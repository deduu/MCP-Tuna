import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { toast } from 'sonner'
import { JudgeConfig } from './JudgeConfig'
import { CriteriaManager } from './CriteriaManager'
import { SingleEvalForm } from './SingleEvalForm'
import { CompareForm } from './CompareForm'
import { MetricsTable } from './MetricsTable'
import { BrowsePathField } from './BrowsePathField'
import { isVlmDatasetRow } from '@/lib/evaluation-multimodal'

type EvalMode = 'single' | 'rubric' | 'batch' | 'compare'

const MODES: { value: EvalMode; label: string }[] = [
  { value: 'single', label: 'Single' },
  { value: 'rubric', label: 'Rubric' },
  { value: 'batch', label: 'Batch' },
  { value: 'compare', label: 'Compare' },
]

export function JudgeTab() {
  const [configOpen, setConfigOpen] = useState(false)
  const [criteriaOpen, setCriteriaOpen] = useState(false)
  const [mode, setMode] = useState<EvalMode>('single')
  const [batchPath, setBatchPath] = useState('')
  const { mutateAsync: executeTool, isPending: batchPending, data: batchResult } = useToolExecution()

  async function handleBatch() {
    if (!batchPath.trim()) {
      toast.error('Dataset path is required')
      return
    }
    try {
      const loaded = await executeTool({ toolName: 'dataset.load', args: { file_path: batchPath } })
      const payload = loaded as Record<string, unknown>
      const testData = Array.isArray(payload.data_points)
        ? (payload.data_points as Array<Record<string, unknown>>)
        : []
      const toolName = testData.some((row) => isVlmDatasetRow(row))
        ? 'judge.evaluate_vlm_batch'
        : 'judge.evaluate_batch'
      await executeTool({ toolName, args: { test_data: testData } })
      toast.success('Batch evaluation complete')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Batch evaluation failed')
    }
  }

  return (
    <div className="space-y-6">
      {/* Collapsible Config */}
      <Card>
        <button
          type="button"
          className="flex w-full items-center gap-2 p-4 text-left font-medium cursor-pointer"
          onClick={() => setConfigOpen((v) => !v)}
        >
          {configOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          Judge Configuration
        </button>
        {configOpen && (
          <CardContent>
            <JudgeConfig />
          </CardContent>
        )}
      </Card>

      {/* Collapsible Criteria */}
      <Card>
        <button
          type="button"
          className="flex w-full items-center gap-2 p-4 text-left font-medium cursor-pointer"
          onClick={() => setCriteriaOpen((v) => !v)}
        >
          {criteriaOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          Evaluation Criteria
        </button>
        {criteriaOpen && (
          <CardContent>
            <CriteriaManager />
          </CardContent>
        )}
      </Card>

      {/* Mode selector */}
      <div className="flex gap-1 rounded-lg bg-muted p-1 w-fit">
        {MODES.map((m) => (
          <button
            key={m.value}
            type="button"
            onClick={() => setMode(m.value)}
            className={cn(
              'px-4 py-1.5 text-sm font-medium rounded-md transition-colors cursor-pointer',
              mode === m.value
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Form based on mode */}
      {(mode === 'single' || mode === 'rubric') && <SingleEvalForm mode={mode} />}
      {mode === 'compare' && <CompareForm />}
      {mode === 'batch' && (
        <Card>
          <CardContent className="pt-6 space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Dataset Path</label>
              <BrowsePathField
                value={batchPath}
                onChange={setBatchPath}
                placeholder="/path/to/dataset.jsonl"
                allowFiles
                allowDirectories={false}
                preferredRootIds={['workspace', 'uploads', 'output']}
                helperText="Browse a dataset file for batch judge evaluation. VLM rows with canonical messages will use the multimodal judge tool automatically."
              />
            </div>
            <Button onClick={handleBatch} disabled={batchPending}>
              {batchPending ? 'Running...' : 'Batch Judge'}
            </Button>
            {batchResult && (
              <MetricsTable data={batchResult as Record<string, unknown>} />
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
