import { useMemo } from 'react'
import { useNavigate } from 'react-router'
import { useToolRegistry } from '@/api/hooks/useToolRegistry'
import { NAMESPACES, getNamespaceFromToolName } from '@/lib/tool-registry'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Input } from '@/components/ui/input'
import { Search } from 'lucide-react'
import { useState } from 'react'

export function ToolExplorerPage() {
  const { data: tools, isLoading } = useToolRegistry()
  const navigate = useNavigate()
  const [search, setSearch] = useState('')

  const toolCounts = useMemo(() => {
    if (!tools) return {}
    const counts: Record<string, number> = {}
    for (const tool of tools) {
      const ns = getNamespaceFromToolName(tool.name)
      counts[ns] = (counts[ns] ?? 0) + 1
    }
    return counts
  }, [tools])

  const filteredNamespaces = useMemo(() => {
    if (!search.trim()) return NAMESPACES
    const q = search.toLowerCase()
    return NAMESPACES.filter(
      (ns) =>
        ns.label.toLowerCase().includes(q) ||
        ns.description.toLowerCase().includes(q) ||
        ns.id.includes(q),
    )
  }, [search])

  const vlmToolCount = useMemo(
    () => (tools ?? []).filter((tool) => tool.name.includes('vlm')).length,
    [tools],
  )

  if (isLoading) {
    return (
      <div className="mx-auto w-full max-w-5xl space-y-4">
        <Skeleton className="h-9 w-72" />
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto w-full max-w-5xl space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        <span>Browse MCP namespaces.</span>
        {vlmToolCount > 0 && <Badge variant="outline">{vlmToolCount} multimodal tools available</Badge>}
        <span>Tools with `messages` inputs accept structured text plus image blocks.</span>
      </div>
      <div className="relative w-72">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Filter namespaces..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-9"
        />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {filteredNamespaces.map((ns) => (
          <Card
            key={ns.id}
            className="cursor-pointer hover:border-primary/30 transition-all group"
            onClick={() => navigate(`/tools/${ns.id}`)}
          >
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <span
                  className="h-3 w-3 rounded-full"
                  style={{ backgroundColor: ns.color }}
                />
                <span className="font-semibold text-sm group-hover:text-primary transition-colors">
                  {ns.label}
                </span>
                <span className="ml-auto text-xs font-mono text-muted-foreground">
                  {toolCounts[ns.id] ?? 0}
                </span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {ns.description}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
