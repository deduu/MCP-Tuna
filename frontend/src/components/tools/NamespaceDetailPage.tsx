import { useMemo } from 'react'
import { useParams, useNavigate, Link } from 'react-router'
import { useToolRegistry } from '@/api/hooks/useToolRegistry'
import { NAMESPACE_MAP, getNamespaceFromToolName, getToolShortName } from '@/lib/tool-registry'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { ArrowLeft, Play } from 'lucide-react'

export function NamespaceDetailPage() {
  const { namespace } = useParams<{ namespace: string }>()
  const { data: tools, isLoading } = useToolRegistry()
  const navigate = useNavigate()

  const nsInfo = namespace ? NAMESPACE_MAP[namespace] : undefined

  const nsTools = useMemo(() => {
    if (!tools || !namespace) return []
    return tools.filter((t) => getNamespaceFromToolName(t.name) === namespace)
  }, [tools, namespace])

  if (isLoading) {
    return (
      <div className="space-y-4 max-w-3xl">
        <Skeleton className="h-8 w-48" />
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-20" />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-4 max-w-3xl">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" onClick={() => navigate('/tools')}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <span
          className="h-3 w-3 rounded-full"
          style={{ backgroundColor: nsInfo?.color }}
        />
        <h2 className="font-semibold text-lg">{nsInfo?.label ?? namespace}</h2>
        <Badge variant="secondary">{nsTools.length} tools</Badge>
      </div>

      {nsInfo && (
        <p className="text-sm text-muted-foreground">{nsInfo.description}</p>
      )}

      <div className="space-y-2">
        {nsTools.map((tool) => {
          const shortName = getToolShortName(tool.name)
          const requiredParams = tool.inputSchema?.required ?? []
          const allParams = Object.keys(tool.inputSchema?.properties ?? {})
          const optionalCount = allParams.length - requiredParams.length

          return (
            <Card key={tool.name} className="hover:border-primary/20 transition-colors">
              <CardContent className="p-4 flex items-center gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <Link
                      to={`/tools/${namespace}/${shortName}`}
                      className="font-medium text-sm hover:text-primary transition-colors"
                    >
                      {shortName}
                    </Link>
                    {requiredParams.length > 0 && (
                      <span className="text-[10px] text-muted-foreground font-mono">
                        {requiredParams.length} required
                        {optionalCount > 0 && `, ${optionalCount} optional`}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5 truncate">
                    {tool.description}
                  </p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  className="gap-1.5 shrink-0"
                  onClick={() => navigate(`/tools/${namespace}/${shortName}`)}
                >
                  <Play className="h-3 w-3" />
                  Run
                </Button>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
