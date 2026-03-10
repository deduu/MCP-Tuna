import { useState } from 'react'
import { ChevronDown, Square, Eye } from 'lucide-react'
import type { PipelineJob } from '@/api/hooks/usePipeline'
import { Card, CardContent } from '@/components/ui/card'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { StepIndicator } from './StepIndicator'
import { cn, formatDuration, formatTimeAgo } from '@/lib/utils'

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
  const progress = job.progress

  const isActive = job.status === 'running' || job.status === 'pending'
  const statusVariant = STATUS_VARIANT[job.status] ?? 'secondary'

  const currentStepIndex = job.steps?.indexOf(job.current_step ?? '') ?? -1
  const totalSteps = job.steps?.length ?? 0
  const rawProgress =
    typeof progress?.percent_complete === 'number' ? progress.percent_complete : undefined
  const isTrainingStage = job.current_step === 'train' && (progress?.max_steps ?? 0) > 0
  const stageLabel =
    currentStepIndex >= 0 && totalSteps > 0 ? `Stage ${currentStepIndex + 1} of ${totalSteps}` : null
  const lastUpdateAgo = formatTimeAgo(progress?.last_updated)
  const lastUpdatedMs = progress?.last_updated ? new Date(progress.last_updated).getTime() : NaN
  const isStale =
    isActive && Number.isFinite(lastUpdatedMs) && Date.now() - lastUpdatedMs > 45_000
  const pipelineProgress =
    job.status === 'completed'
      ? 100
      : typeof rawProgress === 'number'
        ? isTrainingStage && totalSteps > 0
          ? ((Math.max(currentStepIndex, 0) + rawProgress / 100) / totalSteps) * 100
          : rawProgress
        : undefined

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
          {stageLabel && (
            <span className="text-xs text-muted-foreground">
              {stageLabel}
            </span>
          )}
          {job.current_step && job.status === 'running' && (
            <span className="text-xs text-muted-foreground">
              Step: {job.current_step}
            </span>
          )}
          {isStale && (
            <Badge variant="warning">No update {lastUpdateAgo ?? 'recently'}</Badge>
          )}
        </div>

        {progress?.status_message && (
          <p className="text-sm text-muted-foreground">
            {progress.status_message}
          </p>
        )}

        {!isTrainingStage && progress?.stage_current != null && progress?.stage_total != null && (
          <div className="text-xs text-muted-foreground">
            {progress.stage_current} / {progress.stage_total}
            {progress.stage_unit ? ` ${progress.stage_unit}${progress.stage_current === 1 ? '' : 's'}` : ''}
          </div>
        )}

        {/* Progress bar */}
        {pipelineProgress != null && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Pipeline progress</span>
              <span className="font-mono">{Math.round(pipelineProgress)}%</span>
            </div>
            <Progress value={pipelineProgress} className="flex-1" />
          </div>
        )}

        {/* Step indicator */}
        {job.steps && job.steps.length > 0 && (
          <StepIndicator
            steps={job.steps}
            currentStep={currentStepIndex >= 0 ? currentStepIndex : 0}
          />
        )}

        {/* Training detail */}
        {isTrainingStage && progress && (
          <div className="space-y-3 rounded-lg border border-border/60 bg-secondary/20 p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-medium">Training stage active</p>
                <p className="text-xs text-muted-foreground">
                  {progress.current_step ?? 0} / {progress.max_steps ?? 0} steps
                  {progress.current_epoch != null && progress.max_epochs != null && (
                    <> • epoch {progress.current_epoch.toFixed(1)} / {progress.max_epochs}</>
                  )}
                  {lastUpdateAgo && <> • updated {lastUpdateAgo}</>}
                </p>
              </div>
              {typeof rawProgress === 'number' && (
                <span className="text-xs font-mono text-muted-foreground">
                  train {Math.round(rawProgress)}%
                </span>
              )}
            </div>

            {typeof rawProgress === 'number' && (
              <Progress value={rawProgress} className="h-2" />
            )}

            <div className="grid grid-cols-2 gap-3 text-xs md:grid-cols-4">
              <div className="rounded-md bg-background/40 p-2">
                <p className="text-muted-foreground">Trainer Step</p>
                <p className="font-mono">
                  {progress.current_step ?? 0} / {progress.max_steps ?? 0}
                </p>
              </div>
              {progress.loss != null && (
                <div className="rounded-md bg-background/40 p-2">
                  <p className="text-muted-foreground">Loss</p>
                  <p className="font-mono">{progress.loss.toFixed(4)}</p>
                </div>
              )}
              {progress.eta_seconds != null && (
                <div className="rounded-md bg-background/40 p-2">
                  <p className="text-muted-foreground">ETA</p>
                  <p className="font-mono">{formatDuration(progress.eta_seconds)}</p>
                </div>
              )}
              {progress.gpu_memory_used_gb != null && progress.gpu_memory_total_gb != null && (
                <div className="rounded-md bg-background/40 p-2">
                  <p className="text-muted-foreground">GPU</p>
                  <p className="font-mono">
                    {progress.gpu_memory_used_gb.toFixed(1)} / {progress.gpu_memory_total_gb.toFixed(1)} GB
                  </p>
                </div>
              )}
            </div>

            {progress.log_history && progress.log_history.length > 0 && (
              <div className="rounded-md bg-background/40 p-2 text-xs text-muted-foreground">
                {(() => {
                  const latest = progress.log_history[progress.log_history.length - 1]
                  return (
                    <>
                      Latest log
                      {latest?.step != null && <> • step {latest.step}</>}
                      {latest?.epoch != null && <> • epoch {Number(latest.epoch).toFixed(2)}</>}
                      {latest?.loss != null && <> • loss {Number(latest.loss).toFixed(4)}</>}
                      {latest?.learning_rate != null && (
                        <> • lr {Number(latest.learning_rate).toExponential(2)}</>
                      )}
                    </>
                  )
                })()}
              </div>
            )}
          </div>
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
