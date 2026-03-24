import { useNavigate } from 'react-router'
import { ArrowRight, Fish, GitBranch, Rocket, Sparkles } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

interface DashboardHeroProps {
  toolCount: number
  vlmToolCount: number
  activeRuns: number
  activeDeployments: number
  readyDatasets: number
  warningCount: number
  completedJobs: number
}

export function DashboardHero({
  toolCount,
  vlmToolCount,
  activeRuns,
  activeDeployments,
  readyDatasets,
  warningCount,
  completedJobs,
}: DashboardHeroProps) {
  const navigate = useNavigate()

  return (
    <Card className="overflow-hidden border-border/70">
      <CardContent className="relative p-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(59,130,246,0.18),_transparent_32%),radial-gradient(circle_at_bottom_right,_rgba(16,185,129,0.16),_transparent_28%)]" />
        <div className="relative flex flex-col gap-6 p-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-4">
            <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-primary/25 bg-primary/12 text-primary">
              <Fish className="h-6 w-6" />
            </div>
            <div className="space-y-2">
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">{toolCount} tools connected</Badge>
                {vlmToolCount > 0 && <Badge variant="secondary">{vlmToolCount} VLM tools ready</Badge>}
                {warningCount > 0 ? (
                  <Badge variant="warning">{warningCount} issues need review</Badge>
                ) : (
                  <Badge variant="success">System ready</Badge>
                )}
              </div>
              <div>
                <h2 className="text-2xl font-semibold tracking-tight">Operate the full model workflow from one screen</h2>
                <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
                  Check system health, pick up recent work, and jump directly into data prep, training,
                  deployment, or evaluation.
                </p>
              </div>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-3 lg:min-w-[420px]">
            <div className="rounded-2xl border border-border/70 bg-background/65 p-4 backdrop-blur-sm">
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Ready to train</p>
              <p className="mt-2 text-2xl font-semibold">{readyDatasets}</p>
              <p className="mt-1 text-xs text-muted-foreground">datasets with rows available</p>
            </div>
            <div className="rounded-2xl border border-border/70 bg-background/65 p-4 backdrop-blur-sm">
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">In progress</p>
              <p className="mt-2 text-2xl font-semibold">{activeRuns + activeDeployments}</p>
              <p className="mt-1 text-xs text-muted-foreground">runs and live deployments</p>
            </div>
            <div className="rounded-2xl border border-border/70 bg-background/65 p-4 backdrop-blur-sm">
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Completed</p>
              <p className="mt-2 text-2xl font-semibold">{completedJobs}</p>
              <p className="mt-1 text-xs text-muted-foreground">finished training jobs retained</p>
            </div>
          </div>
        </div>

        <div className="relative flex flex-wrap items-center gap-3 border-t border-border/70 px-6 py-4">
          <Button onClick={() => navigate('/datasets')} className="gap-2">
            <Sparkles className="h-4 w-4" />
            Prepare Data
          </Button>
          <Button variant="outline" onClick={() => navigate('/training')} className="gap-2">
            <GitBranch className="h-4 w-4" />
            Start Training
          </Button>
          <Button variant="ghost" onClick={() => navigate('/deployments')} className="gap-2">
            <Rocket className="h-4 w-4" />
            Review Deployments
            <ArrowRight className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
