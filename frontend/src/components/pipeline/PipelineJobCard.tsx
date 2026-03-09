import { useState } from 'react'
import { ChevronDown, Square, Eye } from 'lucide-react'
import type { PipelineJob } from '@/api/hooks/usePipeline'
import { Card, CardContent } from '@/components/ui/card'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { StepIndicator } from './StepIndicator'
import { cn } from '@/lib/utils'

interface PipelineJobCardProps {
  job: PipelineJob
  onCancel: (id: string) => void
}

const STATUS_VARIANT: Record<string, BadgeProps['variant']> = {
  running: 'default',
  completed: 'success',
  failed: 'error',
  pending: 'secondary',
  cancelled: 'outline',
}

export function PipelineJobCard({ job, onCancel }: PipelineJobCardProps) {
  const [showResult, setShowResult] = useState(false)
  const result = (job.result as Record<string, unknown> | undefined) ?? null

  const isActive = job.status === 'running' || job.status === 'pending'
  const statusVariant = STATUS_VARIANT[job.status] ?? 'secondary'

  const currentStepIndex = job.steps?.indexOf(job.current_step ?? '') ?? -1

  function handleViewResult() {
    setShowResult((v) => !v)
  }

  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        {/* Header */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-sm text-muted-foreground">
            {job.job_id.slice(0, 8)}
          </span>
          <Badge variant={statusVariant}>{job.status}</Badge>
          {job.current_step && job.status === 'running' && (
            <span className="text-xs text-muted-foreground">
              Step: {job.current_step}
            </span>
          )}
        </div>

        {/* Progress bar */}
        {job.progress != null && (
          <div className="flex items-center gap-3">
            <Progress value={job.progress} className="flex-1" />
            <span className="text-xs font-mono text-muted-foreground w-10 text-right">
              {Math.round(job.progress)}%
            </span>
          </div>
        )}

        {/* Step indicator */}
        {job.steps && job.steps.length > 0 && (
          <StepIndicator
            steps={job.steps}
            currentStep={currentStepIndex >= 0 ? currentStepIndex : 0}
          />
        )}

        {/* Error */}
        {job.error && (
          <p className="text-sm text-red-400">{job.error}</p>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2">
          {job.status === 'completed' && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleViewResult}
              disabled={!result}
              className="gap-1 text-xs"
            >
              <ChevronDown
                className={cn('h-3.5 w-3.5 transition-transform', showResult && 'rotate-180')}
              />
              <Eye className="h-3.5 w-3.5" />
              {showResult ? 'Hide Result' : 'View Result'}
            </Button>
          )}

          {isActive && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onCancel(job.job_id)}
              className="ml-auto gap-1 text-xs text-destructive hover:text-destructive"
            >
              <Square className="h-3.5 w-3.5" />
              Cancel
            </Button>
          )}
        </div>

        {/* Result */}
        {showResult && result && (
          <pre className="max-h-60 overflow-auto rounded-md bg-secondary/50 p-3 text-xs font-mono">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </CardContent>
    </Card>
  )
}
