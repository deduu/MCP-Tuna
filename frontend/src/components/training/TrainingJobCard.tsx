import { useState } from 'react'
import { ChevronDown, Square, RotateCcw, Rocket, Trash2 } from 'lucide-react'
import type { TrainingJob } from '@/api/types'
import { Card, CardContent } from '@/components/ui/card'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { cn, formatDateTime, formatDuration, formatTimeAgo } from '@/lib/utils'
import { getDeployInitialValues } from './deployment-paths'
import { TrainingJobDetail } from './TrainingJobDetail'

interface TrainingJobCardProps {
  job: TrainingJob
  onCancel: (id: string) => void
  onDelete?: (job: TrainingJob) => void
  onRerun?: (job: TrainingJob) => void
  onDeploy?: (job: TrainingJob, type: 'mcp' | 'api') => void
  isDeleting?: boolean
  defaultExpanded?: boolean
}

const TECHNIQUE_STYLE: Record<string, { variant: BadgeProps['variant']; label: string }> = {
  sft: { variant: 'success', label: 'SFT' },
  dpo: { variant: 'default', label: 'DPO' },
  grpo: { variant: 'secondary', label: 'GRPO' },
  kto: { variant: 'warning', label: 'KTO' },
}

const STATUS_VARIANT: Record<string, BadgeProps['variant']> = {
  running: 'default',
  completed: 'success',
  failed: 'error',
  pending: 'secondary',
  cancelled: 'outline',
}

export function TrainingJobCard({
  job,
  onCancel,
  onDelete,
  onRerun,
  onDeploy,
  isDeleting = false,
  defaultExpanded = false,
}: TrainingJobCardProps) {
  const [expanded, setExpanded] = useState(defaultExpanded)
  const techniqueKey = (job.technique ?? job.trainer_type ?? 'unknown').toLowerCase()

  const technique = TECHNIQUE_STYLE[techniqueKey] ?? {
    variant: 'secondary' as const,
    label: techniqueKey.toUpperCase(),
  }
  const statusVariant = STATUS_VARIANT[job.status] ?? 'secondary'
  const isActive = job.status === 'running' || job.status === 'pending'
  const pct = job.progress?.percent_complete ?? 0
  const deployValues = job.status === 'completed' ? getDeployInitialValues(job) : null
  const createdLabel = formatTimeAgo(job.created_at)
  const completedLabel = formatTimeAgo(job.completed_at)

  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        {/* Header row */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-sm text-muted-foreground">
            {job.job_id.slice(0, 8)}
          </span>
          <Badge variant={technique.variant}>{technique.label}</Badge>
          <Badge variant={statusVariant}>{job.status}</Badge>
          <span className="text-sm text-muted-foreground ml-auto hidden sm:inline truncate max-w-48">
            {job.base_model}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
          {createdLabel && <span>Created {createdLabel}</span>}
          {job.completed_at && completedLabel && <span>Finished {completedLabel}</span>}
          {job.elapsed_seconds != null && job.elapsed_seconds > 0 && (
            <span>Duration {formatDuration(job.elapsed_seconds)}</span>
          )}
        </div>

        {/* Progress bar */}
        <div className="flex items-center gap-3">
          <Progress value={pct} className="flex-1" />
          <span className="text-xs font-mono text-muted-foreground w-10 text-right">
            {Math.round(pct)}%
          </span>
        </div>

        {/* Running info row */}
        {job.status === 'running' && job.progress && (
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            {job.progress.eta_seconds != null && (
              <span>ETA: {formatDuration(job.progress.eta_seconds)}</span>
            )}
            {typeof job.progress.loss === 'number' && (
              <span>Loss: {job.progress.loss.toFixed(4)}</span>
            )}
          </div>
        )}

        {job.status === 'completed' && deployValues?.adapterPath && (
          <div className="rounded-md border border-border/70 bg-muted/20 px-3 py-2 text-xs">
            <span className="text-muted-foreground">Adapter:</span>{' '}
            <code className="break-all text-foreground">{deployValues.adapterPath}</code>
          </div>
        )}

        {job.created_at && (
          <div className="rounded-md border border-border/60 bg-secondary/10 px-3 py-2 text-[11px] text-muted-foreground">
            Persistent record retained from {formatDateTime(job.created_at) ?? job.created_at}
          </div>
        )}

        {/* Actions row */}
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded((e) => !e)}
            className="gap-1 text-xs"
          >
            <ChevronDown
              className={cn('h-3.5 w-3.5 transition-transform', expanded && 'rotate-180')}
            />
            {expanded ? 'Collapse' : 'Details'}
          </Button>

          <div className="ml-auto flex gap-1">
            {onRerun && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onRerun(job)}
                className="gap-1 text-xs"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Re-run
              </Button>
            )}
            {job.status === 'completed' && onDeploy && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onDeploy(job, 'mcp')}
                  className="gap-1 text-xs"
                >
                  <Rocket className="h-3.5 w-3.5" />
                  Deploy MCP
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onDeploy(job, 'api')}
                  className="gap-1 text-xs"
                >
                  API
                </Button>
              </>
            )}
            {isActive && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onCancel(job.job_id)}
                disabled={isDeleting}
                className="gap-1 text-xs text-destructive hover:text-destructive"
              >
                <Square className="h-3.5 w-3.5" />
                Cancel
              </Button>
            )}
            {onDelete && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onDelete(job)}
                disabled={isDeleting}
                className="gap-1 text-xs text-destructive hover:text-destructive"
              >
                <Trash2 className="h-3.5 w-3.5" />
                {isDeleting ? 'Removing...' : isActive ? 'Cancel & Delete' : 'Delete'}
              </Button>
            )}
          </div>
        </div>

        {/* Expanded detail */}
        {expanded && <TrainingJobDetail job={job} />}
      </CardContent>
    </Card>
  )
}
