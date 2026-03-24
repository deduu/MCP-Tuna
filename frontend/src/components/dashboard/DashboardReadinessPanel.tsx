import type { SetupCheck } from '@/api/types'
import { AlertTriangle, ShieldCheck, Wrench } from 'lucide-react'
import { useNavigate } from 'react-router'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface NamespaceSummaryItem {
  id: string
  label: string
  count: number
  color: string
}

interface DashboardReadinessPanelProps {
  namespaceSummary: NamespaceSummaryItem[]
  setupIssues: SetupCheck[]
  warnings: string[]
  toolCount: number
  vlmToolCount: number
}

export function DashboardReadinessPanel({
  namespaceSummary,
  setupIssues,
  warnings,
  toolCount,
  vlmToolCount,
}: DashboardReadinessPanelProps) {
  const navigate = useNavigate()
  const priorityChecks = setupIssues.slice(0, 4)
  const topNamespaces = namespaceSummary.slice(0, 6)

  return (
    <div className="space-y-6">
      <Card className="border-border/70">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <ShieldCheck className="h-4 w-4 text-primary" />
            Readiness
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {warnings.length > 0 && (
            <div className="rounded-2xl border border-amber-500/25 bg-amber-500/10 p-4">
              <div className="mb-2 flex items-center gap-2 text-sm font-medium text-amber-300">
                <AlertTriangle className="h-4 w-4" />
                Active warnings
              </div>
              <div className="flex flex-wrap gap-2">
                {warnings.slice(0, 4).map((warning) => (
                  <Badge key={warning} variant="warning" title={warning}>
                    {warning}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {priorityChecks.length > 0 ? (
            <div className="space-y-2">
              <p className="text-sm font-medium">Recommended fixes</p>
              {priorityChecks.map((check) => (
                <button
                  key={check.name}
                  type="button"
                  onClick={() => navigate(check.action_path ?? '/settings')}
                  className="flex w-full items-start justify-between gap-3 rounded-xl border border-border/70 bg-secondary/10 p-3 text-left transition-colors hover:border-primary/35 hover:bg-secondary/20"
                >
                  <div>
                    <p className="text-sm font-medium">{check.name}</p>
                    <p className="mt-1 text-xs text-muted-foreground">{check.detail}</p>
                  </div>
                  <Badge
                    variant={check.status === 'fail' ? 'error' : 'warning'}
                    className="shrink-0"
                  >
                    {check.status}
                  </Badge>
                </button>
              ))}
            </div>
          ) : (
            <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/10 p-4 text-sm text-emerald-300">
              Setup checks are passing. The platform looks ready for normal operation.
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-border/70">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Wrench className="h-4 w-4 text-primary" />
            Platform Snapshot
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-border/70 bg-secondary/10 p-3">
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Tools</p>
              <p className="mt-2 text-2xl font-semibold">{toolCount}</p>
            </div>
            <div className="rounded-xl border border-border/70 bg-secondary/10 p-3">
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">VLM</p>
              <p className="mt-2 text-2xl font-semibold">{vlmToolCount}</p>
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-sm font-medium">Busiest namespaces</p>
            <div className="flex flex-wrap gap-2">
              {topNamespaces.map((namespace) => (
                <Badge key={namespace.id} variant="outline" className="gap-1.5">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: namespace.color }}
                  />
                  {namespace.label}
                  <span className="text-muted-foreground">{namespace.count}</span>
                </Badge>
              ))}
            </div>
          </div>

          <button
            type="button"
            onClick={() => navigate('/tools')}
            className="w-full rounded-xl border border-border/70 bg-secondary/10 px-4 py-3 text-left text-sm transition-colors hover:border-primary/35 hover:bg-secondary/20"
          >
            Open Tool Explorer for the full namespace and tool catalog.
          </button>
        </CardContent>
      </Card>
    </div>
  )
}
