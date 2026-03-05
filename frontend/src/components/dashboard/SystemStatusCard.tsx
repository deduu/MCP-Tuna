import { useSystemResources, useSetupCheck } from '@/api/hooks/useSystemResources'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Cpu, HardDrive, MemoryStick, MonitorCheck } from 'lucide-react'

function ResourceMeter({
  icon: Icon,
  label,
  used,
  total,
  unit,
  color,
}: {
  icon: typeof Cpu
  label: string
  used: number
  total: number
  unit: string
  color: string
}) {
  const pct = total > 0 ? (used / total) * 100 : 0
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Icon className="h-4 w-4" />
          <span>{label}</span>
        </div>
        <span className="font-mono text-xs">
          {used.toFixed(1)} / {total.toFixed(1)} {unit}
        </span>
      </div>
      <Progress value={pct} color={color} />
    </div>
  )
}

export function SystemStatusCard() {
  const { data: resources, isLoading: resLoading, error: resError } = useSystemResources()
  const { data: setup, isLoading: setupLoading } = useSetupCheck()

  if (resLoading) {
    return (
      <Card>
        <CardHeader><CardTitle>System Status</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (resError) {
    return (
      <Card>
        <CardHeader><CardTitle>System Status</CardTitle></CardHeader>
        <CardContent>
          <p className="text-sm text-destructive">
            Failed to connect to MCP gateway. Is it running on port 8000?
          </p>
        </CardContent>
      </Card>
    )
  }

  const gpu = resources?.gpu
  const ram = resources?.ram
  const disk = resources?.disk

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MonitorCheck className="h-4 w-4" />
          System Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {gpu?.available ? (
          <ResourceMeter
            icon={Cpu}
            label={gpu.name ?? 'GPU'}
            used={gpu.vram_used_gb ?? 0}
            total={gpu.vram_total_gb ?? 0}
            unit="GB VRAM"
            color="var(--color-ns-finetune)"
          />
        ) : (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Cpu className="h-4 w-4" />
            <span>No GPU detected (CPU mode)</span>
          </div>
        )}

        <ResourceMeter
          icon={MemoryStick}
          label="RAM"
          used={ram?.used_gb ?? 0}
          total={ram?.total_gb ?? 0}
          unit="GB"
          color="var(--color-ns-generate)"
        />

        <ResourceMeter
          icon={HardDrive}
          label="Disk"
          used={disk?.used_gb ?? 0}
          total={disk?.total_gb ?? 0}
          unit="GB"
          color="var(--color-ns-dataset)"
        />

        {!setupLoading && setup?.checks && (
          <div className="pt-2 border-t space-y-1">
            <p className="text-xs text-muted-foreground font-medium mb-2">Prerequisites</p>
            <div className="flex flex-wrap gap-1.5">
              {setup.checks.map((check) => (
                <Badge
                  key={check.name}
                  variant={check.status === 'pass' ? 'success' : check.status === 'warn' ? 'warning' : 'error'}
                  title={check.detail}
                >
                  {check.name}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
