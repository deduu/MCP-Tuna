import { useMemo } from 'react'
import { usePipelineJobs, useCancelPipeline } from '@/api/hooks/usePipeline'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { includesTrainingStage } from '@/lib/training-progress'
import { PipelineJobCard } from './PipelineJobCard'
import { toast } from 'sonner'

export function PipelineJobTracker() {
  const { data: jobs, isLoading, error } = usePipelineJobs()
  const cancel = useCancelPipeline()
  const visibleJobs = useMemo(
    () => (jobs ?? []).filter((job) => !includesTrainingStage(job.steps)),
    [jobs],
  )

  function handleCancel(jobId: string) {
    cancel.mutate(jobId, {
      onSuccess: () => toast.success('Pipeline job cancelled'),
      onError: (err) => toast.error(`Cancel failed: ${err.message}`),
    })
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h2 className="text-lg font-semibold">Pipeline Jobs</h2>
        <Badge variant="secondary">{visibleJobs.length}</Badge>
      </div>

      {isLoading && (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-32 w-full rounded-xl" />
          ))}
        </div>
      )}

      {!isLoading && error && (
        <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-4 text-sm text-red-300">
          Unable to load pipeline jobs from the gateway. The backend may be down or restarting.
          <div className="mt-1 font-mono text-xs text-red-200/80">
            {error.message}
          </div>
        </div>
      )}

      {!isLoading && !error && visibleJobs.length === 0 && (
        <p className="text-sm text-muted-foreground py-8 text-center">
          No non-training pipeline jobs here. Pipeline runs with a training stage are shown on the Training page.
        </p>
      )}

      {visibleJobs.map((job) => (
        <PipelineJobCard key={job.job_id} job={job} onCancel={handleCancel} />
      ))}
    </div>
  )
}
