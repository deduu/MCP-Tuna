import { usePipelineJobs, useCancelPipeline } from '@/api/hooks/usePipeline'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { PipelineJobCard } from './PipelineJobCard'
import { toast } from 'sonner'

export function PipelineJobTracker() {
  const { data: jobs, isLoading } = usePipelineJobs()
  const cancel = useCancelPipeline()

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
        {jobs && <Badge variant="secondary">{jobs.length}</Badge>}
      </div>

      {isLoading && (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-32 w-full rounded-xl" />
          ))}
        </div>
      )}

      {!isLoading && (!jobs || jobs.length === 0) && (
        <p className="text-sm text-muted-foreground py-8 text-center">
          No async pipeline jobs available. Current workflow tools execute synchronously.
        </p>
      )}

      {jobs &&
        jobs.map((job) => (
          <PipelineJobCard key={job.job_id} job={job} onCancel={handleCancel} />
        ))}
    </div>
  )
}
