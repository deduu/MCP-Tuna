import { Badge } from '@/components/ui/badge'
import { CheckCircle2, LoaderCircle } from 'lucide-react'

interface PendingDeploymentQueueItem {
  requestId: string
  name: string
  modelPath: string
  adapterPath?: string
  type: 'mcp' | 'api'
  modality: 'text' | 'vision-language'
  port: number
  phase: 'starting' | 'syncing'
}

interface PendingDeploymentQueueProps {
  items: PendingDeploymentQueueItem[]
}

function basename(value: string): string {
  const normalized = value.trim().replace(/\\/g, '/')
  return normalized.split('/').pop() || normalized
}

export function PendingDeploymentQueue({ items }: PendingDeploymentQueueProps) {
  if (items.length === 0) {
    return null
  }

  const startingCount = items.filter((item) => item.phase === 'starting').length

  return (
    <section className="relative rounded-2xl border border-primary/25 bg-primary/5 p-4 shadow-[0_16px_36px_rgba(59,130,246,0.08)]">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <LoaderCircle className="h-4 w-4 animate-spin text-primary" />
            <p className="text-sm font-medium">Deployment starting in the background</p>
            <Badge variant="outline">
              {items.length} {items.length === 1 ? 'runtime' : 'runtimes'}
            </Badge>
          </div>
          <p className="max-w-3xl text-xs leading-relaxed text-muted-foreground">
            The form is closed now, so you can keep working elsewhere in MCP Tuna while the runtime
            starts and syncs into the inventory.
          </p>
        </div>

        <p className="text-xs text-muted-foreground">
          {startingCount > 0
            ? `${startingCount} still launching`
            : 'Runtime ready, waiting for inventory refresh'}
        </p>
      </div>

      <div className="mt-4 grid gap-3 xl:grid-cols-2">
        {items.map((item) => {
          const displayName = item.name.trim() || basename(item.modelPath)
          const isSyncing = item.phase === 'syncing'

          return (
            <div
              key={item.requestId}
              className="rounded-xl border border-border/90 bg-card/80 p-4 shadow-sm shadow-black/20"
            >
              <div className="flex flex-wrap items-center gap-2">
                <Badge
                  className={
                    item.type === 'mcp'
                      ? 'border-transparent bg-[var(--color-ns-host)]/20 text-[var(--color-ns-host)]'
                      : 'border-transparent bg-primary/20 text-primary'
                  }
                >
                  {item.type === 'mcp' ? 'MCP' : 'API'}
                </Badge>
                <Badge variant="outline">
                  {item.modality === 'vision-language' ? 'Vision-Language' : 'Text'}
                </Badge>
                <Badge variant="outline">Port {item.port}</Badge>
              </div>

              <p className="mt-3 truncate text-sm font-medium">{displayName}</p>
              <p className="mt-1 truncate text-xs text-muted-foreground">{item.modelPath}</p>
              {item.adapterPath && (
                <p className="mt-1 truncate text-xs text-muted-foreground">
                  Adapter: {item.adapterPath}
                </p>
              )}

              <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
                {isSyncing ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                ) : (
                  <LoaderCircle className="h-3.5 w-3.5 animate-spin text-primary" />
                )}
                <span>
                  {isSyncing
                    ? 'Runtime is ready. Updating the inventory now.'
                    : 'Starting runtime. You can keep using other pages while it loads.'}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
