import { useMemo } from 'react'
import { DashboardHero } from './DashboardHero'
import { DashboardReadinessPanel } from './DashboardReadinessPanel'
import { DashboardRecentActivity } from './DashboardRecentActivity'
import { DashboardStatsGrid } from './DashboardStatsGrid'
import { DashboardWorkflowSection } from './DashboardWorkflowSection'
import { SystemStatusCard } from './SystemStatusCard'
import { useToolCount, useToolRegistry } from '@/api/hooks/useToolRegistry'
import { useDatasets } from '@/api/hooks/useDatasets'
import { useTrainingJobs } from '@/api/hooks/useTraining'
import { useDeployments } from '@/api/hooks/useDeployments'
import { usePipelineJobs } from '@/api/hooks/usePipeline'
import { useSetupCheck, useSystemHealth } from '@/api/hooks/useSystemResources'
import { NAMESPACE_MAP, getNamespaceFromToolName } from '@/lib/tool-registry'
import { includesTrainingStage } from '@/lib/training-progress'

export function DashboardPage() {
  const { data: tools } = useToolRegistry()
  const { toolCount } = useToolCount()
  const { data: datasets = [] } = useDatasets()
  const { data: jobs = [] } = useTrainingJobs()
  const { data: deployments = [] } = useDeployments()
  const { data: pipelineJobs = [] } = usePipelineJobs()
  const { data: setup } = useSetupCheck()
  const { data: health } = useSystemHealth()

  const toolsByNamespace = useMemo(() => {
    const counts: Record<string, number> = {}
    if (!tools) return counts

    for (const tool of tools) {
      const namespace = getNamespaceFromToolName(tool.name)
      counts[namespace] = (counts[namespace] ?? 0) + 1
    }

    return counts
  }, [tools])

  const trainingPipelineJobs = pipelineJobs.filter((job) => includesTrainingStage(job.steps))
  const activeTrainingJobs = jobs.filter((job) => job.status === 'running' || job.status === 'pending').length
  const activePipelineJobs = trainingPipelineJobs.filter(
    (job) => job.status === 'running' || job.status === 'pending',
  ).length
  const activeRuns = activeTrainingJobs + activePipelineJobs
  const activeDeployments = deployments.filter((deployment) => deployment.status === 'running').length
  const completedJobs = jobs.filter((job) => job.status === 'completed').length
  const readyDatasets = datasets.filter((dataset) => dataset.row_count > 0).length
  const setupIssues = setup?.checks.filter((check) => check.status !== 'pass') ?? []
  const warningCount = (health?.warnings.length ?? 0) + setupIssues.length
  const vlmToolCount = tools?.filter((tool) => tool.name.includes('vlm')).length ?? 0
  const reviewPath =
    setupIssues.length === 1 && setupIssues[0].action_path
      ? setupIssues[0].action_path
      : warningCount > 0
        ? '/settings#diagnostics'
        : '/settings'
  const reviewDetail =
    setupIssues.length === 1
      ? setupIssues[0].detail ?? `${setupIssues[0].name} needs attention`
      : health?.warnings.length === 1 && setupIssues.length === 0
        ? health.warnings[0]
        : 'Warnings or setup checks need attention'

  const namespaceSummary = Object.entries(toolsByNamespace)
    .map(([namespace, count]) => ({
      id: namespace,
      count,
      label: NAMESPACE_MAP[namespace]?.label ?? namespace,
      color: NAMESPACE_MAP[namespace]?.color ?? 'var(--color-muted-foreground)',
    }))
    .sort((a, b) => b.count - a.count)

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <DashboardHero
        toolCount={toolCount}
        vlmToolCount={vlmToolCount}
        activeRuns={activeRuns}
        activeDeployments={activeDeployments}
        readyDatasets={readyDatasets}
        warningCount={warningCount}
        completedJobs={completedJobs}
      />

      <DashboardStatsGrid
        activeRuns={activeRuns}
        activeDeployments={activeDeployments}
        readyDatasets={readyDatasets}
        warningCount={warningCount}
        completedJobs={completedJobs}
        reviewPath={reviewPath}
        reviewDetail={reviewDetail}
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.5fr)_minmax(320px,1fr)]">
        <div className="space-y-6">
          <DashboardRecentActivity
            jobs={jobs}
            pipelineJobs={pipelineJobs}
            deployments={deployments}
            datasets={datasets}
          />
          <SystemStatusCard />
        </div>

        <div className="space-y-6">
          <DashboardWorkflowSection />
          <DashboardReadinessPanel
            namespaceSummary={namespaceSummary}
            setupIssues={setupIssues}
            warnings={health?.warnings ?? []}
            toolCount={toolCount}
            vlmToolCount={vlmToolCount}
          />
        </div>
      </div>
    </div>
  )
}
