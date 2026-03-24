import { useState } from 'react'
import { Link } from 'react-router'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useSetupCheck, useSystemConfig, useSystemHealth } from '@/api/hooks/useSystemResources'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import type { MCPToolResult, RecommendResult } from '@/api/types'
import { Stethoscope, ChevronDown, ChevronUp, Play, HardDrive, Sparkles } from 'lucide-react'

interface CheckItem {
  name: string
  status: 'pass' | 'warn' | 'fail'
  detail?: string
}

interface PreflightResult extends MCPToolResult {
  can_run?: boolean
  estimated_vram_gb?: number
  available_vram_gb?: number
  headroom_gb?: number
  recommendations?: string[]
  warnings?: string[]
}

function formatGb(value?: number) {
  return typeof value === 'number' ? `${value.toFixed(2)} GB` : 'n/a'
}

function buildPreflightChecks(result: PreflightResult | undefined): CheckItem[] {
  if (!result) return []

  return [
    {
      name: 'Can Run',
      status: result.can_run ? 'pass' : 'warn',
      detail: result.can_run ? 'Configuration fits available VRAM' : 'Configuration is likely to OOM',
    },
    {
      name: 'Estimated VRAM',
      status: 'pass',
      detail: formatGb(result.estimated_vram_gb),
    },
    {
      name: 'Available VRAM',
      status: typeof result.available_vram_gb === 'number' && result.available_vram_gb > 0 ? 'pass' : 'warn',
      detail: formatGb(result.available_vram_gb),
    },
    {
      name: 'Headroom',
      status:
        typeof result.headroom_gb === 'number'
          ? result.headroom_gb >= 1
            ? 'pass'
            : result.headroom_gb >= 0
              ? 'warn'
              : 'fail'
          : 'warn',
      detail: formatGb(result.headroom_gb),
    },
  ]
}

function CollapsibleResult({
  label,
  isOpen,
  onToggle,
  children,
}: {
  label: string
  isOpen: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="flex items-center justify-between w-full px-3 py-2 text-sm text-muted-foreground hover:bg-secondary/50 transition-colors cursor-pointer"
      >
        <span>{label}</span>
        {isOpen ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
      </button>
      {isOpen && <div className="px-3 pb-3 border-t">{children}</div>}
    </div>
  )
}

function CheckList({ checks }: { checks: CheckItem[] }) {
  return (
    <div className="space-y-1.5 pt-2">
      {checks.map((check) => (
        <div key={check.name} className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">{check.name}</span>
          <div className="flex items-center gap-2">
            {check.detail && (
              <span className="text-xs text-muted-foreground font-mono">{check.detail}</span>
            )}
            <Badge
              variant={
                check.status === 'pass' ? 'success' : check.status === 'warn' ? 'warning' : 'error'
              }
            >
              {check.status}
            </Badge>
          </div>
        </div>
      ))}
    </div>
  )
}

export function DiagnosticsSection() {
  const { data: setup, isLoading: setupLoading } = useSetupCheck()
  const { data: config, isLoading: configLoading } = useSystemConfig()
  const { data: health, error: healthError, isLoading: healthLoading } = useSystemHealth()

  const preflight = useToolExecution()
  const diskPreflight = useToolExecution()
  const recommendations = useToolExecution()

  const [openSections, setOpenSections] = useState<Record<string, boolean>>({})
  const baseModel = config?.config?.finetuning?.base_model

  const toggle = (key: string) =>
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }))

  const runPreflight = () => {
    if (!baseModel) return

    preflight.mutate(
      { toolName: 'system.preflight_check', args: { model_name: baseModel } },
      { onSuccess: () => setOpenSections((prev) => ({ ...prev, preflight: true })) },
    )
  }

  const runDisk = () => {
    diskPreflight.mutate(
      { toolName: 'system.disk_preflight', args: {} },
      { onSuccess: () => setOpenSections((prev) => ({ ...prev, disk: true })) },
    )
  }

  const runRecommendations = () => {
    recommendations.mutate(
      { toolName: 'validate.recommend_models', args: { use_case: 'general' } },
      { onSuccess: () => setOpenSections((prev) => ({ ...prev, recommendations: true })) },
    )
  }

  const preflightData = preflight.data as PreflightResult | undefined
  const recommendData = recommendations.data as RecommendResult | undefined

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Stethoscope className="h-4 w-4" />
          System Diagnostics
        </CardTitle>
        <CardDescription>Run checks to verify your environment is properly configured</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <p className="text-sm font-medium">Live Warnings</p>
          {healthLoading ? (
            <p className="text-xs text-muted-foreground">Loading live system warnings...</p>
          ) : healthError ? (
            <p className="text-xs text-destructive">{healthError.message}</p>
          ) : health?.warnings?.length ? (
            <div className="flex flex-wrap gap-1.5">
              {health.warnings.map((warning) => (
                <Badge key={warning} variant="warning" title={warning}>
                  {warning}
                </Badge>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No live system warnings</p>
          )}
        </div>

        {/* Setup Check - auto-loaded */}
        <div className="space-y-2">
          <p className="text-sm font-medium">Setup Check</p>
          {setupLoading ? (
            <p className="text-xs text-muted-foreground">Loading prerequisites...</p>
          ) : setup?.checks ? (
            <div className="flex flex-wrap gap-1.5">
              {setup.checks.map((check) => (
                check.action_path ? (
                  <Link key={check.name} to={check.action_path} title={check.detail} className="transition-opacity hover:opacity-85">
                    <Badge
                      variant={
                        check.status === 'pass'
                          ? 'success'
                          : check.status === 'warn'
                            ? 'warning'
                            : 'error'
                      }
                    >
                      {check.name}
                    </Badge>
                  </Link>
                ) : (
                  <Badge
                    key={check.name}
                    variant={
                      check.status === 'pass'
                        ? 'success'
                        : check.status === 'warn'
                          ? 'warning'
                          : 'error'
                    }
                    title={check.detail}
                  >
                    {check.name}
                  </Badge>
                )
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">Unable to load setup checks</p>
          )}
        </div>

        {/* Preflight Check */}
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground">
            Base model:{' '}
            <span className="font-mono">
              {baseModel ?? (configLoading ? 'Loading configuration...' : 'Unavailable')}
            </span>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={runPreflight}
            disabled={preflight.isPending || !baseModel}
          >
            <Play className="h-3.5 w-3.5" />
            Run Preflight
          </Button>
          {preflight.error && (
            <p className="text-sm text-destructive">{preflight.error.message}</p>
          )}
          {preflight.data && (
            <CollapsibleResult
              label="Preflight Results"
              isOpen={openSections.preflight ?? false}
              onToggle={() => toggle('preflight')}
            >
              <CheckList checks={buildPreflightChecks(preflightData)} />
              {!!preflightData?.recommendations?.length && (
                <div className="pt-3 text-sm">
                  <p className="font-medium">Recommendations</p>
                  <ul className="list-disc pl-5 pt-1 text-muted-foreground">
                    {preflightData.recommendations.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
              {!!preflightData?.warnings?.length && (
                <div className="pt-3 text-sm">
                  <p className="font-medium">Warnings</p>
                  <ul className="list-disc pl-5 pt-1 text-muted-foreground">
                    {preflightData.warnings.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
            </CollapsibleResult>
          )}
        </div>

        {/* Disk Preflight */}
        <div className="space-y-2">
          <Button
            variant="outline"
            size="sm"
            onClick={runDisk}
            disabled={diskPreflight.isPending}
          >
            <HardDrive className="h-3.5 w-3.5" />
            Disk Preflight
          </Button>
          {diskPreflight.error && (
            <p className="text-sm text-destructive">{diskPreflight.error.message}</p>
          )}
          {diskPreflight.data && (
            <CollapsibleResult
              label="Disk Analysis"
              isOpen={openSections.disk ?? false}
              onToggle={() => toggle('disk')}
            >
              <pre className="mt-2 text-xs font-mono text-muted-foreground whitespace-pre-wrap overflow-x-auto">
                {JSON.stringify(diskPreflight.data, null, 2)}
              </pre>
            </CollapsibleResult>
          )}
        </div>

        {/* AI Recommendations */}
        <div className="space-y-2">
          <Button
            variant="outline"
            size="sm"
            onClick={runRecommendations}
            disabled={recommendations.isPending}
          >
            <Sparkles className="h-3.5 w-3.5" />
            AI Recommendations
          </Button>
          {recommendations.error && (
            <p className="text-sm text-destructive">{recommendations.error.message}</p>
          )}
          {recommendData && (
            <CollapsibleResult
              label="Recommended Models"
              isOpen={openSections.recommendations ?? false}
              onToggle={() => toggle('recommendations')}
            >
              {recommendData.recommendations?.length ? (
                <div className="space-y-2 pt-2">
                  {recommendData.recommendations.map((item) => (
                    <div key={item.model_id} className="rounded-md border px-3 py-2">
                      <div className="text-sm font-medium">{item.model_id}</div>
                      <div className="text-xs text-muted-foreground">
                        {item.size} | {item.memory}
                      </div>
                      <p className="pt-1 text-sm text-muted-foreground">{item.description}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <pre className="mt-2 text-xs font-mono text-muted-foreground whitespace-pre-wrap overflow-x-auto leading-relaxed">
                  {JSON.stringify(recommendData, null, 2)}
                </pre>
              )}
            </CollapsibleResult>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
