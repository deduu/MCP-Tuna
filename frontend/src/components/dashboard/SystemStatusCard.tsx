import type { ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router'
import { useSystemHealth, useSystemResources, useSetupCheck } from '@/api/hooks/useSystemResources'
import type { GPUInfo } from '@/api/types'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import {
  AlertTriangle,
  Cpu,
  HardDrive,
  MemoryStick,
  MonitorCheck,
  RefreshCw,
  Rocket,
  Server,
  Sparkles,
} from 'lucide-react'
import { toast } from 'sonner'

function healthBadgeVariant(status?: 'green' | 'yellow' | 'red') {
  if (status === 'green') return 'success'
  if (status === 'yellow') return 'warning'
  if (status === 'red') return 'error'
  return 'secondary'
}

function healthLabel(status?: 'green' | 'yellow' | 'red') {
  if (status === 'green') return 'Healthy'
  if (status === 'yellow') return 'Busy'
  if (status === 'red') return 'Needs Attention'
  return 'Unknown'
}

function formatTimestamp(timestamp: number) {
  if (!timestamp) return 'Waiting for data'
  return new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function InlineMeta({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full border border-border/60 bg-secondary/30 px-2 py-0.5 text-[11px] text-muted-foreground">
      {children}
    </span>
  )
}

function ResourceMeter({
  icon: Icon,
  label,
  used,
  total,
  unit,
  color,
  meta,
}: {
  icon: typeof Cpu
  label: string
  used: number
  total: number
  unit: string
  color: string
  meta?: ReactNode
}) {
  const pct = total > 0 ? (used / total) * 100 : 0
  return (
    <div className="space-y-2 rounded-xl border border-border/60 bg-secondary/10 p-3">
      <div className="flex items-center justify-between gap-3 text-sm">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Icon className="h-4 w-4" />
          <span>{label}</span>
        </div>
        <span className="font-mono text-xs">
          {used.toFixed(1)} / {total.toFixed(1)} {unit}
        </span>
      </div>
      <Progress value={pct} color={color} />
      {meta ? <div className="flex flex-wrap gap-1.5">{meta}</div> : null}
    </div>
  )
}

function GPUInventory({ gpus }: { gpus: GPUInfo[] }) {
  if (gpus.length <= 1) return null

  return (
    <div className="space-y-2 rounded-xl border border-border/60 bg-secondary/10 p-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
          GPU Inventory
        </p>
        <Badge variant="outline">{gpus.length} detected</Badge>
      </div>
      <div className="space-y-2">
        {gpus.map((device) => {
          const used = device.vram_used_gb ?? 0
          const total = device.vram_total_gb ?? 0
          const reserved = device.vram_reserved_gb ?? 0
          return (
            <div key={device.index ?? device.name} className="rounded-lg border border-border/60 p-3">
              <div className="mb-2 flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium">{device.name ?? `GPU ${device.index ?? '?'}`}</p>
                  <p className="text-xs text-muted-foreground">Device {device.index ?? 0}</p>
                </div>
                <span className="font-mono text-xs">
                  {used.toFixed(1)} / {total.toFixed(1)} GB
                </span>
              </div>
              <Progress value={used} max={Math.max(total, 1)} color="var(--color-ns-finetune)" />
              <div className="mt-2 flex flex-wrap gap-1.5">
                <InlineMeta>Reserved {reserved.toFixed(1)} GB</InlineMeta>
                <InlineMeta>CUDA {device.cuda_version ?? '?'}</InlineMeta>
                <InlineMeta>CC {device.compute_capability ?? '?'}</InlineMeta>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function SystemStatusCard() {
  const queryClient = useQueryClient()
  const resourcesQuery = useSystemResources()
  const healthQuery = useSystemHealth()
  const { data: setup, isLoading: setupLoading } = useSetupCheck()
  const clearGpuCache = useToolExecution()

  const { data: resources, isLoading: resLoading, error: resError, dataUpdatedAt: resourcesUpdatedAt } = resourcesQuery
  const { data: health, error: healthError, isFetching: healthFetching, dataUpdatedAt: healthUpdatedAt } = healthQuery

  if (resLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (resError || !resources) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-destructive">
            Failed to connect to MCP gateway. Is it running on port 8000?
          </p>
        </CardContent>
      </Card>
    )
  }

  const gpu = resources.gpu
  const gpus = resources.gpus?.length ? resources.gpus : (gpu?.available ? [gpu] : [])
  const ram = resources.ram
  const disk = resources.disk
  const lastUpdated = Math.max(resourcesUpdatedAt, healthUpdatedAt)
  const warnings = health?.warnings ?? []
  const healthStatus = healthError ? 'yellow' : health?.status
  const healthText = healthError ? 'Health Unavailable' : healthLabel(health?.status)

  const refreshAll = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['system', 'resources'] }),
      queryClient.invalidateQueries({ queryKey: ['system', 'health'] }),
      queryClient.invalidateQueries({ queryKey: ['system', 'setup'] }),
    ])
  }

  const runGpuRecovery = () => {
    clearGpuCache.mutate(
      { toolName: 'system.clear_gpu_cache', args: {} },
      {
        onSuccess: async (result) => {
          await refreshAll()
          const stats = result.memory_stats as Record<string, number> | undefined
          const allocated = typeof stats?.allocated_gb === 'number' ? stats.allocated_gb.toFixed(1) : '?'
          const reserved = typeof stats?.reserved_gb === 'number' ? stats.reserved_gb.toFixed(1) : '?'
          toast.success(`Freed process GPU cache. Allocated ${allocated} GB, reserved ${reserved} GB.`)
        },
        onError: (error) => {
          const message = error.message.includes('Unknown tool: system.clear_gpu_cache')
            ? 'GPU recovery failed: restart the gateway on port 8002 to load the new system.clear_gpu_cache tool.'
            : `GPU recovery failed: ${error.message}`
          toast.error(message)
        },
      },
    )
  }

  const gpuUsed = gpu?.vram_used_gb ?? 0
  const gpuTotal = gpu?.vram_total_gb ?? 0
  const gpuReserved = gpu?.vram_reserved_gb ?? 0

  return (
    <Card>
      <CardHeader className="space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <MonitorCheck className="h-4 w-4" />
              System Status
            </CardTitle>
            <CardDescription className="mt-1">
              Updated {formatTimestamp(lastUpdated)}
            </CardDescription>
          </div>
          <div className="flex flex-wrap justify-end gap-2">
            <Badge variant={healthBadgeVariant(healthStatus)}>{healthText}</Badge>
            {(resources.gpu_count ?? gpus.length) > 1 && (
              <Badge variant="outline">{resources.gpu_count ?? gpus.length} GPUs</Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2 rounded-xl border border-border/60 bg-secondary/10 p-3">
          <Badge variant="outline" className="gap-1.5">
            <Rocket className="h-3 w-3" />
            {health?.active_training_jobs ?? 0} jobs
          </Badge>
          <Badge variant="outline" className="gap-1.5">
            <Server className="h-3 w-3" />
            {health?.active_deployments ?? 0} deployments
          </Badge>
          <div className="ml-auto flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={() => void refreshAll()} disabled={healthFetching}>
              <RefreshCw className={`h-3.5 w-3.5 ${healthFetching ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={runGpuRecovery}
              disabled={!gpu?.available || clearGpuCache.isPending}
            >
              <Sparkles className="h-3.5 w-3.5" />
              {clearGpuCache.isPending ? 'Freeing VRAM...' : 'Free VRAM'}
            </Button>
          </div>
        </div>

        {gpu?.available ? (
          <ResourceMeter
            icon={Cpu}
            label={gpu.name ?? 'GPU'}
            used={gpuUsed}
            total={gpuTotal}
            unit="GB VRAM"
            color="var(--color-ns-finetune)"
            meta={
              <>
                <InlineMeta>Reserved {gpuReserved.toFixed(1)} GB</InlineMeta>
                <InlineMeta>CUDA {gpu.cuda_version ?? '?'}</InlineMeta>
                <InlineMeta>CC {gpu.compute_capability ?? '?'}</InlineMeta>
              </>
            }
          />
        ) : (
          <div className="flex items-center gap-2 rounded-xl border border-border/60 bg-secondary/10 p-3 text-sm text-muted-foreground">
            <Cpu className="h-4 w-4" />
            <span>No GPU detected (CPU mode)</span>
          </div>
        )}

        <GPUInventory gpus={gpus} />

        <ResourceMeter
          icon={MemoryStick}
          label="RAM"
          used={ram.used_gb ?? 0}
          total={ram.total_gb ?? 0}
          unit="GB"
          color="var(--color-ns-generate)"
          meta={
            <>
              <InlineMeta>Free {ram.free_gb.toFixed(1)} GB</InlineMeta>
              <InlineMeta>{ram.percent_used.toFixed(0)}% used</InlineMeta>
            </>
          }
        />

        <ResourceMeter
          icon={HardDrive}
          label="Disk"
          used={disk.used_gb ?? 0}
          total={disk.total_gb ?? 0}
          unit="GB"
          color="var(--color-ns-dataset)"
          meta={
            <>
              <InlineMeta>Free {disk.free_gb.toFixed(1)} GB</InlineMeta>
              <InlineMeta>{disk.output_dir}</InlineMeta>
            </>
          }
        />

        {warnings.length > 0 && (
          <div className="space-y-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3">
            <div className="flex items-center gap-2 text-sm font-medium text-amber-300">
              <AlertTriangle className="h-4 w-4" />
              Live Warnings
            </div>
            <div className="flex flex-wrap gap-1.5">
              {warnings.map((warning) => (
                <Badge key={warning} variant="warning" title={warning}>
                  {warning}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {healthError && (
          <div className="space-y-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3">
            <div className="flex items-center gap-2 text-sm font-medium text-amber-300">
              <AlertTriangle className="h-4 w-4" />
              Health Check Unavailable
            </div>
            <p className="text-sm text-amber-200/90">
              {healthError.message}
            </p>
          </div>
        )}

        {!setupLoading && setup?.checks && (
          <div className="space-y-2 rounded-xl border border-border/60 bg-secondary/10 p-3">
            <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
              Prerequisites
            </p>
            <div className="flex flex-wrap gap-1.5">
              {setup.checks.map((check) => (
                check.action_path ? (
                  <Link key={check.name} to={check.action_path} title={check.detail} className="transition-opacity hover:opacity-85">
                    <Badge
                      variant={check.status === 'pass' ? 'success' : check.status === 'warn' ? 'warning' : 'error'}
                    >
                      {check.name}
                    </Badge>
                  </Link>
                ) : (
                  <Badge
                    key={check.name}
                    variant={check.status === 'pass' ? 'success' : check.status === 'warn' ? 'warning' : 'error'}
                    title={check.detail}
                  >
                    {check.name}
                  </Badge>
                )
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
