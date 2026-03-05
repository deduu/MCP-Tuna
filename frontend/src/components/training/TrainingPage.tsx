import { useState, useMemo } from 'react'
import { Cpu, Plus } from 'lucide-react'
import { toast } from 'sonner'
import { useTrainingJobs, useAvailableModels, useCancelTraining } from '@/api/hooks/useTraining'
import { useSystemResources } from '@/api/hooks/useSystemResources'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabList, Tab, TabPanel } from '@/components/ui/tabs'
import { NewTrainingPanel } from './NewTrainingPanel'
import { TrainingJobCard } from './TrainingJobCard'

export function TrainingPage() {
  const [panelOpen, setPanelOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('active')

  const { data: jobs = [], isLoading: jobsLoading } = useTrainingJobs()
  const { data: resources } = useSystemResources()
  const { data: models = [] } = useAvailableModels()
  const cancelTraining = useCancelTraining()

  const activeJobs = useMemo(
    () => jobs.filter((j) => j.status === 'running' || j.status === 'pending'),
    [jobs],
  )
  const completedJobs = useMemo(
    () => jobs.filter((j) => j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled'),
    [jobs],
  )

  const filteredJobs = activeTab === 'active' ? activeJobs : activeTab === 'completed' ? completedJobs : jobs

  function handleCancel(jobId: string) {
    cancelTraining.mutate(jobId, {
      onSuccess: () => toast.success('Training job cancelled'),
      onError: (err) => toast.error(`Cancel failed: ${err.message}`),
    })
  }

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Cpu className="h-6 w-6 text-primary" />
        <h1 className="text-2xl font-bold">Training</h1>
        {activeJobs.length > 0 && (
          <Badge variant="default">{activeJobs.length} active</Badge>
        )}
      </div>

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
          />

          {panelOpen && (
            <Button variant="ghost" size="sm" onClick={() => setPanelOpen(false)} className="text-xs">
              Collapse
            </Button>
          )}

          {/* Job tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabList>
              <Tab value="active">Active ({activeJobs.length})</Tab>
              <Tab value="completed">Completed ({completedJobs.length})</Tab>
              <Tab value="all">All ({jobs.length})</Tab>
            </TabList>

            {['active', 'completed', 'all'].map((tabValue) => (
              <TabPanel key={tabValue} value={tabValue}>
                {jobsLoading ? (
                  <div className="space-y-3">
                    <Skeleton className="h-28 w-full rounded-xl" />
                    <Skeleton className="h-28 w-full rounded-xl" />
                  </div>
                ) : filteredJobs.length === 0 ? (
                  <div className="text-sm text-muted-foreground py-8 text-center">
                    No {tabValue === 'all' ? '' : tabValue} training jobs
                  </div>
                ) : (
                  <div className="space-y-3">
                    {filteredJobs.map((job) => (
                      <TrainingJobCard
                        key={job.job_id}
                        job={job}
                        onCancel={handleCancel}
                      />
                    ))}
                  </div>
                )}
              </TabPanel>
            ))}
          </Tabs>
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
                    <li key={m} className="text-xs font-mono text-muted-foreground truncate" title={m}>
                      {m}
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
