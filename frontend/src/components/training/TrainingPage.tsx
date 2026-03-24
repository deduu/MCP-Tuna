import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Cpu, Plus } from 'lucide-react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'
import type { TrainingJob } from '@/api/types'
import { useTrainingJobs, useAvailableModels, useCancelTraining, useRemoveTrainingJob } from '@/api/hooks/useTraining'
import { usePipelineJobs, useCancelPipeline, useRemovePipelineJob, type PipelineJob } from '@/api/hooks/usePipeline'
import { useSystemResources } from '@/api/hooks/useSystemResources'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { includesTrainingStage } from '@/lib/training-progress'
import { NewTrainingPanel } from './NewTrainingPanel'
import { getDeployInitialValues } from './deployment-paths'
import { TrainingHistoryDetailPanel } from './TrainingHistoryDetailPanel'
import { TrainingHistoryList } from './TrainingHistoryList'
import {
  buildTrainingHistoryEntries,
  filterTrainingHistoryEntries,
  getTrainingHistoryFilterCounts,
  paginateTrainingHistoryEntries,
  type TrainingHistoryFilter,
} from './training-history'

const HISTORY_FETCH_LIMIT = 200
const HISTORY_PAGE_SIZE = 20

export function TrainingPage() {
  const [panelOpen, setPanelOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<TrainingHistoryFilter>('active')
  const [selectedModelPath, setSelectedModelPath] = useState('')
  const [removingJobId, setRemovingJobId] = useState<string | null>(null)
  const [removingPipelineJobId, setRemovingPipelineJobId] = useState<string | null>(null)
  const [historyQuery, setHistoryQuery] = useState('')
  const [historyPage, setHistoryPage] = useState(1)
  const [selectedHistoryKey, setSelectedHistoryKey] = useState<string | null>(null)
  const navigate = useNavigate()

  const { data: jobs = [], isLoading: jobsLoading, error: jobsError } = useTrainingJobs(HISTORY_FETCH_LIMIT)
  const { data: pipelineJobs = [], isLoading: pipelineJobsLoading } = usePipelineJobs(HISTORY_FETCH_LIMIT)
  const { data: resources } = useSystemResources()
  const { data: models = [] } = useAvailableModels()
  const cancelTraining = useCancelTraining()
  const removeTrainingJob = useRemoveTrainingJob()
  const cancelPipeline = useCancelPipeline()
  const removePipelineJob = useRemovePipelineJob()

  const activeJobs = useMemo(
    () => jobs.filter((j) => j.status === 'running' || j.status === 'pending'),
    [jobs],
  )
  const pipelineTrainingJobs = useMemo(
    () => pipelineJobs.filter((job) => includesTrainingStage(job.steps)),
    [pipelineJobs],
  )
  const activePipelineJobs = useMemo(
    () => pipelineTrainingJobs.filter((job) => job.status === 'running' || job.status === 'pending'),
    [pipelineTrainingJobs],
  )
  const completedJobs = useMemo(
    () => jobs.filter((j) => j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled'),
    [jobs],
  )
  const failedJobs = useMemo(
    () => jobs.filter((j) => j.status === 'failed'),
    [jobs],
  )
  const totalActiveRuns = activeJobs.length + activePipelineJobs.length
  const historyEntries = useMemo(
    () => buildTrainingHistoryEntries(jobs, pipelineTrainingJobs),
    [jobs, pipelineTrainingJobs],
  )
  const historyCounts = useMemo(
    () => getTrainingHistoryFilterCounts(historyEntries),
    [historyEntries],
  )
  const filteredHistory = useMemo(
    () => filterTrainingHistoryEntries(historyEntries, activeTab, historyQuery),
    [historyEntries, activeTab, historyQuery],
  )
  const pagedHistory = useMemo(
    () => paginateTrainingHistoryEntries(filteredHistory, historyPage, HISTORY_PAGE_SIZE),
    [filteredHistory, historyPage],
  )
  const selectedEntry = useMemo(
    () => pagedHistory.items.find((entry) => entry.key === selectedHistoryKey) ?? pagedHistory.items[0] ?? null,
    [pagedHistory.items, selectedHistoryKey],
  )
  const historyMayBeTruncated =
    jobs.length >= HISTORY_FETCH_LIMIT || pipelineTrainingJobs.length >= HISTORY_FETCH_LIMIT

  useEffect(() => {
    setHistoryPage(1)
  }, [activeTab, historyQuery])

  useEffect(() => {
    if (historyPage !== pagedHistory.page) {
      setHistoryPage(pagedHistory.page)
    }
  }, [historyPage, pagedHistory.page])

  useEffect(() => {
    const nextKey = selectedEntry?.key ?? null
    if (selectedHistoryKey !== nextKey) {
      setSelectedHistoryKey(nextKey)
    }
  }, [selectedEntry, selectedHistoryKey])

  function handleCancel(jobId: string) {
    cancelTraining.mutate(jobId, {
      onSuccess: () => toast.success('Training job cancelled'),
      onError: (err) => toast.error(`Cancel failed: ${err.message}`),
    })
  }

  function handleDeploy(job: TrainingJob, type: 'mcp' | 'api') {
    const initialValues = getDeployInitialValues(job)
    if (!initialValues?.modelPath) {
      toast.error('Could not determine deployment paths for this training job')
      return
    }

    navigate('/deployments', {
      state: {
        openDeployDialog: true,
        deployDialogType: type,
        deployInitialValues: initialValues,
      },
    })
  }

  function handleDelete(job: TrainingJob) {
    setRemovingJobId(job.job_id)
    removeTrainingJob.mutate(
      { jobId: job.job_id, status: job.status },
      {
        onSuccess: () => {
          toast.success(
            job.status === 'running' || job.status === 'pending'
              ? 'Training job cancelled and deleted'
              : 'Training job deleted',
          )
        },
        onError: (err) => toast.error(`Delete failed: ${err.message}`),
        onSettled: () => setRemovingJobId(null),
      },
    )
  }

  function handleCancelPipeline(jobId: string) {
    cancelPipeline.mutate(jobId, {
      onSuccess: () => toast.success('Pipeline job cancelled'),
      onError: (err) => toast.error(`Cancel failed: ${err.message}`),
    })
  }

  function handleDeletePipeline(job: PipelineJob) {
    setRemovingPipelineJobId(job.job_id)
    removePipelineJob.mutate(
      { jobId: job.job_id, status: job.status },
      {
        onSuccess: () => {
          toast.success(
            job.status === 'running' || job.status === 'pending'
              ? 'Workflow job cancelled and deleted'
              : 'Workflow job deleted',
          )
        },
        onError: (err) => toast.error(`Delete failed: ${err.message}`),
        onSettled: () => setRemovingPipelineJobId(null),
      },
    )
  }

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Cpu className="h-6 w-6 text-primary" />
        <h1 className="text-2xl font-bold">Training</h1>
        {totalActiveRuns > 0 && (
          <Badge variant="default">{totalActiveRuns} active</Badge>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        <Badge variant="outline">{historyEntries.length} loaded runs</Badge>
        <Badge variant="secondary">{pipelineTrainingJobs.length} pipeline runs</Badge>
        <Badge variant="outline">{jobs.length} direct jobs</Badge>
        <Badge variant="success">{completedJobs.filter((job) => job.status === 'completed').length} completed</Badge>
        {failedJobs.length > 0 && <Badge variant="error">{failedJobs.length} failed</Badge>}
        <span>History is loaded from persisted backend state after refresh and restart.</span>
      </div>

      {jobsError && (
        <Card className="border-amber-500/30 bg-amber-500/10">
          <CardContent className="flex items-start gap-3 p-4 text-sm text-amber-200/90">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />
            <div>
              <p className="font-medium text-amber-300">Training jobs unavailable</p>
              <p>{jobsError.message}</p>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Main content */}
        <div className="flex-1 min-w-0 space-y-6">
          {/* New Training toggle */}
          {!panelOpen && (
            <Button variant="outline" onClick={() => setPanelOpen(true)} className="gap-2">
              <Plus className="h-4 w-4" />
              New Training Job
            </Button>
          )}

          {/* New Training Panel */}
          <NewTrainingPanel
            open={panelOpen}
            onToggle={() => setPanelOpen((o) => !o)}
            onSubmit={() => setPanelOpen(false)}
            modelPath={selectedModelPath}
            onModelPathChange={setSelectedModelPath}
          />

          {panelOpen && (
            <Button variant="ghost" size="sm" onClick={() => setPanelOpen(false)} className="text-xs">
              Collapse
            </Button>
          )}

          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">Run Browser</h2>
                <p className="text-sm text-muted-foreground">
                  Browse direct jobs and pipeline-backed runs from one list, then inspect the selected run on the right.
                </p>
              </div>
              {historyMayBeTruncated && (
                <Badge variant="warning">Newest {HISTORY_FETCH_LIMIT} records loaded per source</Badge>
              )}
            </div>

            {jobsLoading || pipelineJobsLoading ? (
              <div className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
                <Skeleton className="h-[520px] rounded-xl" />
                <Skeleton className="h-[520px] rounded-xl" />
              </div>
            ) : (
              <div className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
                <TrainingHistoryList
                  entries={pagedHistory.items}
                  selectedKey={selectedEntry?.key ?? null}
                  onSelect={setSelectedHistoryKey}
                  filter={activeTab}
                  onFilterChange={setActiveTab}
                  counts={historyCounts}
                  query={historyQuery}
                  onQueryChange={setHistoryQuery}
                  page={pagedHistory.page}
                  totalPages={pagedHistory.totalPages}
                  totalItems={pagedHistory.totalItems}
                  startIndex={pagedHistory.startIndex}
                  endIndex={pagedHistory.endIndex}
                  onPageChange={setHistoryPage}
                  isTruncated={historyMayBeTruncated}
                />
                <TrainingHistoryDetailPanel
                  entry={selectedEntry}
                  onCancelTraining={handleCancel}
                  onDeleteTraining={handleDelete}
                  onDeployTraining={handleDeploy}
                  onCancelPipeline={handleCancelPipeline}
                  onDeletePipeline={handleDeletePipeline}
                  removingTrainingJobId={removingJobId}
                  removingPipelineJobId={removingPipelineJobId}
                />
              </div>
            )}
          </div>
        </div>

        {/* Side panel */}
        <div className="w-full lg:w-72 space-y-4 shrink-0">
          {/* GPU/RAM resources */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">System Resources</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {resources?.gpu.available ? (
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">GPU</span>
                    <span className="font-mono">
                      {resources.gpu.vram_used_gb?.toFixed(1) ?? '?'} /{' '}
                      {resources.gpu.vram_total_gb?.toFixed(1) ?? '?'} GB
                    </span>
                  </div>
                  <Progress
                    value={resources.gpu.vram_used_gb ?? 0}
                    max={resources.gpu.vram_total_gb ?? 1}
                  />
                  <p className="text-[10px] text-muted-foreground truncate">
                    {resources.gpu.name}
                  </p>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">No GPU detected</p>
              )}

              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">RAM</span>
                  <span className="font-mono">
                    {resources?.ram.used_gb.toFixed(1) ?? '?'} /{' '}
                    {resources?.ram.total_gb.toFixed(1) ?? '?'} GB
                  </span>
                </div>
                <Progress
                  value={resources?.ram.used_gb ?? 0}
                  max={resources?.ram.total_gb ?? 1}
                />
              </div>
            </CardContent>
          </Card>

          {/* Available models */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Available Models</CardTitle>
            </CardHeader>
            <CardContent>
              {models.length === 0 ? (
                <p className="text-xs text-muted-foreground">No models found</p>
              ) : (
                <ul className="space-y-1 max-h-48 overflow-y-auto">
                  {models.map((m) => (
                    <li key={m}>
                      <button
                        type="button"
                        title={m}
                        onClick={() => {
                          setSelectedModelPath(m)
                          setPanelOpen(true)
                        }}
                        className={cn(
                          'w-full rounded-md border px-2 py-2 text-left text-xs font-mono transition-colors cursor-pointer',
                          'hover:border-primary/40 hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                          selectedModelPath === m
                            ? 'border-primary bg-primary/10 text-foreground'
                            : 'border-transparent text-muted-foreground',
                        )}
                      >
                        <span className="block truncate">{m}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
