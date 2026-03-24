import type { DatasetInfo, Deployment, TrainingJob } from '@/api/types'
import type { PipelineJob } from '@/api/hooks/usePipeline'
import { ArrowUpRight, Database, GitBranch, Rocket, Rows3, Sparkles } from 'lucide-react'
import { useNavigate } from 'react-router'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatDateTime, formatTimeAgo } from '@/lib/utils'
import {
  formatDatasetTitle,
  formatDeploymentTitle,
  formatJobTitle,
  formatPipelineTitle,
  formatTechniqueLabel,
  getPathTail,
  getStatusBadgeVariant,
} from './dashboard-utils'

interface DashboardRecentActivityProps {
  jobs: TrainingJob[]
  pipelineJobs: PipelineJob[]
  deployments: Deployment[]
  datasets: DatasetInfo[]
}

interface ActivitySection {
  title: string
  description: string
  emptyLabel: string
  path: string
  icon: typeof Sparkles
  items: Array<{
    id: string
    title: string
    subtitle: string
    meta: string
    status?: string
    time?: string | null
  }>
}

function ActivityCard({
  title,
  description,
  emptyLabel,
  path,
  icon: Icon,
  items,
}: ActivitySection) {
  const navigate = useNavigate()

  return (
    <Card className="border-border/70">
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2 text-base">
              <Icon className="h-4 w-4 text-primary" />
              {title}
            </CardTitle>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
          <button
            type="button"
            onClick={() => navigate(path)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-border/70 text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
            aria-label={`Open ${title}`}
          >
            <ArrowUpRight className="h-4 w-4" />
          </button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {items.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border/70 px-4 py-6 text-sm text-muted-foreground">
            {emptyLabel}
          </div>
        ) : (
          items.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => navigate(path)}
              className="flex w-full flex-col gap-2 rounded-xl border border-border/70 bg-secondary/10 p-4 text-left transition-colors hover:border-primary/35 hover:bg-secondary/20"
            >
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{item.title}</p>
                  <p className="truncate text-xs text-muted-foreground">{item.subtitle}</p>
                </div>
                {item.status ? (
                  <Badge variant={getStatusBadgeVariant(item.status)}>
                    {item.status}
                  </Badge>
                ) : null}
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <span>{item.meta}</span>
                {item.time ? <span>{item.time}</span> : null}
              </div>
            </button>
          ))
        )}
      </CardContent>
    </Card>
  )
}

function sortByRecent<T>(items: T[], getTimestamp: (item: T) => string | undefined) {
  return [...items].sort((a, b) => {
    const left = Date.parse(getTimestamp(a) ?? '')
    const right = Date.parse(getTimestamp(b) ?? '')
    return (Number.isFinite(right) ? right : 0) - (Number.isFinite(left) ? left : 0)
  })
}

export function DashboardRecentActivity({
  jobs,
  pipelineJobs,
  deployments,
  datasets,
}: DashboardRecentActivityProps) {
  const recentJobs = sortByRecent(jobs, (job) => job.started_at ?? job.created_at)
    .slice(0, 3)
    .map((job) => ({
      id: job.job_id,
      title: formatJobTitle(job),
      subtitle: getPathTail(job.dataset_path ?? job.output_dir),
      meta: job.progress?.percent_complete != null
        ? `${Math.round(job.progress.percent_complete)}% complete`
        : 'Progress unavailable',
      status: job.status,
      time: formatTimeAgo(job.started_at ?? job.created_at),
    }))

  const recentPipelines = sortByRecent(pipelineJobs, (job) => job.started_at ?? job.created_at)
    .slice(0, 3)
    .map((job) => ({
      id: job.job_id,
      title: formatPipelineTitle(job),
      subtitle: job.steps?.length ? `${job.steps.length} steps configured` : 'Workflow job',
      meta: job.progress?.percent_complete != null
        ? `${Math.round(job.progress.percent_complete)}% complete`
        : formatTechniqueLabel(job.current_step),
      status: job.status,
      time: formatTimeAgo(job.started_at ?? job.created_at),
    }))

  const recentDatasets = sortByRecent(datasets, (dataset) => dataset.modified_at)
    .slice(0, 3)
    .map((dataset) => ({
      id: dataset.file_path,
      title: formatDatasetTitle(dataset),
      subtitle: dataset.technique ? formatTechniqueLabel(dataset.technique) : dataset.format.toUpperCase(),
      meta: `${dataset.row_count.toLocaleString()} rows`,
      status: undefined,
      time: formatDateTime(dataset.modified_at),
    }))

  const recentDeployments = sortByRecent(deployments, (deployment) => deployment.updated_at ?? deployment.created_at)
    .slice(0, 3)
    .map((deployment) => ({
      id: deployment.deployment_id,
      title: formatDeploymentTitle(deployment),
      subtitle: deployment.type === 'mcp' ? 'MCP deployment' : 'API deployment',
      meta: deployment.endpoint,
      status: deployment.status,
      time: formatTimeAgo(deployment.updated_at ?? deployment.created_at),
    }))

  const sections: ActivitySection[] = [
    {
      title: 'Training jobs',
      description: 'Resume jobs in progress or review recent runs.',
      emptyLabel: 'No training jobs yet. Start with a model and dataset when you are ready.',
      path: '/training',
      icon: Sparkles,
      items: recentJobs,
    },
    {
      title: 'Pipelines',
      description: 'Keep full workflow runs visible without leaving the dashboard.',
      emptyLabel: 'No pipeline runs yet. Use the pipeline builder for end-to-end automation.',
      path: '/pipeline',
      icon: GitBranch,
      items: recentPipelines,
    },
    {
      title: 'Datasets',
      description: 'Your latest training inputs stay close to the next step.',
      emptyLabel: 'No datasets loaded yet. Import, generate, or build one from documents.',
      path: '/datasets',
      icon: Database,
      items: recentDatasets,
    },
    {
      title: 'Deployments',
      description: 'Check serving status and jump into live model details.',
      emptyLabel: 'No deployments yet. Deploy a trained model as MCP or API when ready.',
      path: '/deployments',
      icon: Rocket,
      items: recentDeployments,
    },
  ]

  return (
    <section className="space-y-4">
      <div className="flex items-center gap-2">
        <Rows3 className="h-4 w-4 text-primary" />
        <h3 className="text-lg font-semibold">Recent Activity</h3>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {sections.map((section) => (
          <ActivityCard key={section.title} {...section} />
        ))}
      </div>
    </section>
  )
}
