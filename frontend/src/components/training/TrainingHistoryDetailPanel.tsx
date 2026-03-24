import type { TrainingJob } from '@/api/types'
import type { PipelineJob } from '@/api/hooks/usePipeline'
import { Card, CardContent } from '@/components/ui/card'
import { PipelineJobCard } from '@/components/pipeline/PipelineJobCard'
import { TrainingJobCard } from './TrainingJobCard'
import type { TrainingHistoryEntry } from './training-history'

interface TrainingHistoryDetailPanelProps {
  entry: TrainingHistoryEntry | null
  onCancelTraining: (id: string) => void
  onDeleteTraining: (job: TrainingJob) => void
  onDeployTraining: (job: TrainingJob, type: 'mcp' | 'api') => void
  onCancelPipeline: (id: string) => void
  onDeletePipeline: (job: PipelineJob) => void
  removingTrainingJobId?: string | null
  removingPipelineJobId?: string | null
}

export function TrainingHistoryDetailPanel({
  entry,
  onCancelTraining,
  onDeleteTraining,
  onDeployTraining,
  onCancelPipeline,
  onDeletePipeline,
  removingTrainingJobId = null,
  removingPipelineJobId = null,
}: TrainingHistoryDetailPanelProps) {
  if (!entry) {
    return (
      <Card className="border-border/70">
        <CardContent className="flex min-h-[420px] items-center justify-center p-6 text-sm text-muted-foreground">
          Select a run from the left to inspect progress, outputs, and actions.
        </CardContent>
      </Card>
    )
  }

  if (entry.kind === 'training') {
    return (
      <TrainingJobCard
        key={entry.key}
        job={entry.job}
        onCancel={onCancelTraining}
        onDelete={onDeleteTraining}
        onDeploy={onDeployTraining}
        isDeleting={removingTrainingJobId === entry.job.job_id}
        defaultExpanded
      />
    )
  }

  return (
    <PipelineJobCard
      key={entry.key}
      job={entry.job}
      onCancel={onCancelPipeline}
      onDelete={onDeletePipeline}
      isDeleting={removingPipelineJobId === entry.job.job_id}
    />
  )
}
