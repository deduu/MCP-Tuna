import type { LucideIcon } from 'lucide-react'
import { AlertTriangle, CheckCircle2, Database, Rocket, Workflow } from 'lucide-react'
import { useNavigate } from 'react-router'
import { Card, CardContent } from '@/components/ui/card'

interface DashboardStatsGridProps {
  activeRuns: number
  activeDeployments: number
  readyDatasets: number
  warningCount: number
  completedJobs: number
  reviewPath: string
  reviewDetail: string
}

interface StatCardData {
  label: string
  value: string
  detail: string
  icon: LucideIcon
  path: string
  color: string
}

function StatCard({ label, value, detail, icon: Icon, path, color }: StatCardData) {
  const navigate = useNavigate()

  return (
    <Card
      className="cursor-pointer border-border/70 transition-colors hover:border-primary/40"
      onClick={() => navigate(path)}
    >
      <CardContent className="flex items-start justify-between gap-3 p-5">
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="text-3xl font-semibold tracking-tight">{value}</p>
          <p className="text-xs text-muted-foreground">{detail}</p>
        </div>
        <div
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-border/60"
          style={{ backgroundColor: `color-mix(in srgb, ${color} 16%, transparent)` }}
        >
          <Icon className="h-5 w-5" style={{ color }} />
        </div>
      </CardContent>
    </Card>
  )
}

export function DashboardStatsGrid({
  activeRuns,
  activeDeployments,
  readyDatasets,
  warningCount,
  completedJobs,
  reviewPath,
  reviewDetail,
}: DashboardStatsGridProps) {
  const cards: StatCardData[] = [
    {
      label: 'Active Work',
      value: activeRuns.toString(),
      detail: activeRuns > 0 ? 'Training and pipeline runs in flight' : 'No training runs active right now',
      icon: Workflow,
      path: '/training',
      color: 'var(--color-ns-workflow)',
    },
    {
      label: 'Deployments',
      value: activeDeployments.toString(),
      detail: activeDeployments > 0 ? 'Endpoints currently serving traffic' : 'No live model deployments',
      icon: Rocket,
      path: '/deployments',
      color: 'var(--color-ns-host)',
    },
    {
      label: 'Datasets Ready',
      value: readyDatasets.toString(),
      detail: readyDatasets > 0 ? 'Datasets available for downstream work' : 'Import or generate your first dataset',
      icon: Database,
      path: '/datasets',
      color: 'var(--color-ns-dataset)',
    },
    {
      label: 'Needs Review',
      value: warningCount.toString(),
      detail: warningCount > 0 ? reviewDetail : `${completedJobs} training jobs completed cleanly`,
      icon: warningCount > 0 ? AlertTriangle : CheckCircle2,
      path: reviewPath,
      color: warningCount > 0 ? 'var(--color-ns-evaluate)' : 'var(--color-ns-host)',
    },
  ]

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {cards.map((card) => (
        <StatCard key={card.label} {...card} />
      ))}
    </div>
  )
}
