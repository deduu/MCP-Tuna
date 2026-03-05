import { SystemStatusCard } from './SystemStatusCard'
import { QuickActions } from './QuickActions'
import { useToolRegistry } from '@/api/hooks/useToolRegistry'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Wrench, Fish } from 'lucide-react'
import { NAMESPACE_MAP, getNamespaceFromToolName } from '@/lib/tool-registry'

export function DashboardPage() {
  const { data: tools } = useToolRegistry()

  const toolsByNamespace: Record<string, number> = {}
  if (tools) {
    for (const tool of tools) {
      const ns = getNamespaceFromToolName(tool.name)
      toolsByNamespace[ns] = (toolsByNamespace[ns] ?? 0) + 1
    }
  }

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="flex items-center gap-3">
        <Fish className="h-8 w-8 text-primary" />
        <div>
          <h2 className="text-xl font-bold">MCP Tuna</h2>
          <p className="text-sm text-muted-foreground">
            End-to-end LLM fine-tuning platform
          </p>
        </div>
        {tools && (
          <Badge variant="secondary" className="ml-auto">
            <Wrench className="h-3 w-3 mr-1" />
            {tools.length} tools
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <SystemStatusCard />
        </div>
        <div className="lg:col-span-2 space-y-6">
          <div>
            <h3 className="text-sm font-semibold mb-3">Quick Actions</h3>
            <QuickActions />
          </div>
        </div>
      </div>

      {tools && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Tool Namespaces</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {Object.entries(toolsByNamespace).map(([ns, count]) => {
                const info = NAMESPACE_MAP[ns]
                return (
                  <Badge
                    key={ns}
                    variant="outline"
                    className="gap-1.5 py-1 px-2.5"
                  >
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: info?.color }}
                    />
                    {info?.label ?? ns}
                    <span className="text-muted-foreground">{count}</span>
                  </Badge>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
