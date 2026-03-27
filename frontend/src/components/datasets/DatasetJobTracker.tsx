import { useEffect, useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { TrainingJob } from '@/api/types'
import { useDatasetBlendJobs } from '@/api/hooks/useDatasets'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, Tab, TabList } from '@/components/ui/tabs'
import { cn, formatDuration, formatTimeAgo } from '@/lib/utils'
import { Square, Trash2 } from 'lucide-react'
import { toast } from 'sonner'

type DatasetJobFilter = 'active' | 'finished' | 'all'

const STATUS_VARIANT: Record<string, 'default' | 'success' | 'error' | 'secondary' | 'outline'> = {
  pending: 'secondary',
  running: 'default',
  completed: 'success',
  failed: 'error',
  cancelled: 'outline',
}

function getOutputPath(job: TrainingJob): string | null {
  const result = job.result as Record<string, unknown> | undefined
  const saveResult = result?.save_result as Record<string, unknown> | undefined
  if (typeof saveResult?.file_path === 'string' && saveResult.file_path.trim()) {
    return saveResult.file_path
  }
  const summary = job.config_summary as Record<string, unknown> | undefined
  if (typeof summary?.output_path === 'string' && summary.output_path.trim()) {
    return summary.output_path
  }
  return null
}

function getRowCount(job: TrainingJob): number | null {
  const result = job.result as Record<string, unknown> | undefined
  return typeof result?.count === 'number' ? result.count : null
}

function isActiveStatus(status: TrainingJob['status']) {
  return status === 'running' || status === 'pending'
}

export function DatasetJobTracker() {
  const queryClient = useQueryClient()
  const { data: jobs = [], isLoading, error } = useDatasetBlendJobs(40)
  const { mutateAsync: executeTool } = useToolExecution()
  const [filter, setFilter] = useState<DatasetJobFilter>('active')
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)

  const counts = useMemo(
    () => ({
      active: jobs.filter((job) => isActiveStatus(job.status)).length,
      finished: jobs.filter((job) => !isActiveStatus(job.status)).length,
      all: jobs.length,
    }),
    [jobs],
  )

  const filteredJobs = useMemo(() => {
    if (filter === 'active') {
      return jobs.filter((job) => isActiveStatus(job.status))
    }
    if (filter === 'finished') {
      return jobs.filter((job) => !isActiveStatus(job.status))
    }
    return jobs
  }, [filter, jobs])

  const selectedJob = useMemo(
    () => filteredJobs.find((job) => job.job_id === selectedJobId) ?? filteredJobs[0] ?? null,
    [filteredJobs, selectedJobId],
  )

  useEffect(() => {
    const nextId = selectedJob?.job_id ?? null
    if (selectedJobId !== nextId) {
      setSelectedJobId(nextId)
    }
  }, [selectedJob, selectedJobId])

  async function refreshJobs() {
    await queryClient.invalidateQueries({ queryKey: ['datasets', 'blend-jobs'] })
  }

  async function handleCancel(jobId: string) {
    try {
      await executeTool({
        toolName: 'generate.cancel_hf_blend_job',
        args: { job_id: jobId },
      })
      await refreshJobs()
      toast.success(`Cancellation requested for ${jobId}`)
    } catch (err) {
      toast.error(`Cancel failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleDelete(jobId: string) {
    try {
      await executeTool({
        toolName: 'generate.delete_hf_blend_job',
        args: { job_id: jobId },
      })
      await refreshJobs()
      toast.success(`Deleted ${jobId}`)
    } catch (err) {
      toast.error(`Delete failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">Dataset Jobs</h2>
        <p className="text-sm text-muted-foreground">
          Browse active and finished dataset blend runs, then inspect the selected job on the right.
        </p>
      </div>

      {isLoading ? (
        <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
          <Skeleton className="h-[420px] rounded-xl" />
          <Skeleton className="h-[420px] rounded-xl" />
        </div>
      ) : error ? (
        <Card className="border-border/70">
          <CardContent className="p-4 text-sm text-destructive">{error.message}</CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
          <Card className="border-border/70">
            <CardHeader className="space-y-4 pb-4">
              <div>
                <CardTitle className="text-base">Run List</CardTitle>
                <p className="mt-1 text-sm text-muted-foreground">
                  Select a dataset job to inspect progress, output path, and actions.
                </p>
              </div>
              <Tabs value={filter} onValueChange={(value) => setFilter(value as DatasetJobFilter)}>
                <TabList className="mb-0">
                  <Tab value="active">Active ({counts.active})</Tab>
                  <Tab value="finished">Finished ({counts.finished})</Tab>
                  <Tab value="all">All ({counts.all})</Tab>
                </TabList>
              </Tabs>
            </CardHeader>

            <CardContent className="space-y-3">
              <div className="space-y-2 xl:max-h-[70vh] xl:overflow-y-auto xl:pr-1">
                {filteredJobs.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-border/70 px-4 py-8 text-center text-sm text-muted-foreground">
                    No dataset jobs match the current filter.
                  </div>
                ) : (
                  filteredJobs.map((job) => {
                    const rowCount = getRowCount(job)
                    const outputPath = getOutputPath(job)
                    return (
                      <button
                        key={job.job_id}
                        type="button"
                        onClick={() => setSelectedJobId(job.job_id)}
                        className={cn(
                          'w-full rounded-xl border p-3 text-left transition-colors',
                          selectedJob?.job_id === job.job_id
                            ? 'border-primary bg-primary/10'
                            : 'border-border/70 bg-secondary/10 hover:border-primary/35 hover:bg-secondary/20',
                        )}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0 space-y-2">
                            <div className="flex flex-wrap gap-2">
                              <Badge variant="secondary">HF Blend</Badge>
                              <Badge variant={STATUS_VARIANT[job.status] ?? 'secondary'}>{job.status}</Badge>
                            </div>
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium">{job.job_id}</p>
                              <p className="truncate text-xs text-muted-foreground">
                                {outputPath ?? 'Output path pending'}
                              </p>
                            </div>
                          </div>
                          <span className="text-xs font-mono text-muted-foreground">
                            {Math.round(job.progress?.percent_complete ?? 0)}%
                          </span>
                        </div>
                        <Progress value={job.progress?.percent_complete ?? 0} className="mt-3 h-1.5" />
                        <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                          {rowCount != null && <span>{rowCount.toLocaleString()} rows</span>}
                          {job.created_at && <span>{formatTimeAgo(job.created_at)}</span>}
                        </div>
                      </button>
                    )
                  })
                )}
              </div>
            </CardContent>
          </Card>

          {!selectedJob ? (
            <Card className="border-border/70">
              <CardContent className="flex min-h-[420px] items-center justify-center p-6 text-sm text-muted-foreground">
                Select a dataset job from the left to inspect details.
              </CardContent>
            </Card>
          ) : (
            <Card className="border-border/70">
              <CardContent className="p-4 space-y-4">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-mono text-sm text-muted-foreground">{selectedJob.job_id}</span>
                  <Badge variant="secondary">HF Blend</Badge>
                  <Badge variant={STATUS_VARIANT[selectedJob.status] ?? 'secondary'}>
                    {selectedJob.status}
                  </Badge>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <Progress value={selectedJob.progress?.percent_complete ?? 0} className="flex-1" />
                    <span className="text-xs font-mono text-muted-foreground w-10 text-right">
                      {Math.round(selectedJob.progress?.percent_complete ?? 0)}%
                    </span>
                  </div>
                  {selectedJob.progress?.status_message && (
                    <p className="text-sm text-muted-foreground">
                      {selectedJob.progress.status_message}
                    </p>
                  )}
                </div>

                <div className="grid gap-3 sm:grid-cols-2 text-sm">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Created</p>
                    <p>{selectedJob.created_at ? formatTimeAgo(selectedJob.created_at) : 'Unknown'}</p>
                  </div>
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Duration</p>
                    <p>
                      {selectedJob.elapsed_seconds != null && selectedJob.elapsed_seconds > 0
                        ? formatDuration(selectedJob.elapsed_seconds)
                        : 'In progress'}
                    </p>
                  </div>
                  <div className="sm:col-span-2">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Output Path</p>
                    <p className="break-all text-muted-foreground">{getOutputPath(selectedJob) ?? 'Pending'}</p>
                  </div>
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Rows</p>
                    <p>{getRowCount(selectedJob)?.toLocaleString() ?? 'Pending'}</p>
                  </div>
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Format</p>
                    <p>{String((selectedJob.config_summary?.format as string | undefined) ?? 'jsonl')}</p>
                  </div>
                </div>

                {selectedJob.error && (
                  <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                    {selectedJob.error}
                  </div>
                )}

                <div className="flex flex-wrap justify-end gap-2">
                  {isActiveStatus(selectedJob.status) && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleCancel(selectedJob.job_id)}
                      className="gap-1 text-destructive hover:text-destructive"
                    >
                      <Square className="h-3.5 w-3.5" />
                      Stop Job
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDelete(selectedJob.job_id)}
                    className="gap-1 text-destructive hover:text-destructive"
                    disabled={isActiveStatus(selectedJob.status)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Delete Job
                  </Button>
                </div>

                <div className="rounded-md border border-border/60 bg-secondary/10 p-3">
                  <pre className="max-h-64 overflow-auto text-xs">
                    {JSON.stringify(selectedJob, null, 2)}
                  </pre>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
