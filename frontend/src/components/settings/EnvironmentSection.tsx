import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useSystemResources } from '@/api/hooks/useSystemResources'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Settings2, Cpu, MemoryStick, HardDrive, ExternalLink } from 'lucide-react'
import { Link } from 'react-router'

function ResourceStat({
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

export function EnvironmentSection() {
  const { data: resources, isLoading: resLoading } = useSystemResources()
  const configQuery = useToolExecution()

  const loadConfig = () => {
    configQuery.mutate({ toolName: 'system.config', args: {} })
  }

  const configEntries: [string, unknown][] = configQuery.data
    ? Object.entries(configQuery.data).filter(([key]) => key !== 'success')
    : []

  const gpu = resources?.gpu
  const ram = resources?.ram
  const disk = resources?.disk

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings2 className="h-4 w-4" />
          Environment
        </CardTitle>
        <CardDescription>Configuration and system resources</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Configuration */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium">Configuration</p>
            <Button
              variant="outline"
              size="sm"
              onClick={loadConfig}
              disabled={configQuery.isPending}
            >
              View Configuration
            </Button>
          </div>
          {configQuery.error && (
            <p className="text-sm text-destructive">{configQuery.error.message}</p>
          )}
          {configEntries.length > 0 && (
            <div className="max-h-64 overflow-auto rounded-lg border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-secondary/30">
                    <th className="px-3 py-2 text-left font-medium text-muted-foreground">Key</th>
                    <th className="px-3 py-2 text-left font-medium text-muted-foreground">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {configEntries.map(([key, value]) => (
                    <tr key={key} className="border-b last:border-b-0">
                      <td className="px-3 py-2 font-mono text-xs">{key}</td>
                      <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* System Resources */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium">System Resources</p>
            <Link
              to="/"
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              Full dashboard
              <ExternalLink className="h-3 w-3" />
            </Link>
          </div>

          {resLoading ? (
            <p className="text-xs text-muted-foreground">Loading resources...</p>
          ) : (
            <div className="space-y-3">
              {gpu?.available ? (
                <ResourceStat
                  icon={Cpu}
                  label={gpu.name ?? 'GPU'}
                  used={gpu.vram_used_gb ?? 0}
                  total={gpu.vram_total_gb ?? 0}
                  unit="GB VRAM"
                  color="var(--color-ns-system)"
                />
              ) : (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Cpu className="h-4 w-4" />
                  <span>No GPU detected (CPU mode)</span>
                </div>
              )}

              <ResourceStat
                icon={MemoryStick}
                label="RAM"
                used={ram?.used_gb ?? 0}
                total={ram?.total_gb ?? 0}
                unit="GB"
                color="var(--color-ns-system)"
              />

              <ResourceStat
                icon={HardDrive}
                label="Disk"
                used={disk?.used_gb ?? 0}
                total={disk?.total_gb ?? 0}
                unit="GB"
                color="var(--color-ns-system)"
              />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
