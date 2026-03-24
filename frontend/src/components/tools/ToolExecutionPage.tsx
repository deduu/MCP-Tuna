import { useMemo, useRef } from 'react'
import { useParams, useNavigate } from 'react-router'
import { useToolRegistry } from '@/api/hooks/useToolRegistry'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { NAMESPACE_MAP } from '@/lib/tool-registry'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { ToolParameterForm } from './ToolParameterForm'
import { ToolResultPanel } from './ToolResultPanel'
import { ArrowLeft } from 'lucide-react'

export function ToolExecutionPage() {
  const { namespace, tool: toolShortName } = useParams<{ namespace: string; tool: string }>()
  const { data: tools, isLoading: registryLoading } = useToolRegistry()
  const navigate = useNavigate()
  const execution = useToolExecution()
  const startTimeRef = useRef<number>(0)

  const toolName = `${namespace}.${toolShortName}`
  const tool = useMemo(
    () => tools?.find((t) => t.name === toolName),
    [tools, toolName],
  )
  const supportsMultimodalMessages = Boolean(tool?.inputSchema?.properties?.messages)
  const nsInfo = namespace ? NAMESPACE_MAP[namespace] : undefined

  if (registryLoading) {
    return (
      <div className="mx-auto w-full max-w-2xl space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-48" />
      </div>
    )
  }

  if (!tool) {
    return (
      <div className="mx-auto w-full max-w-2xl">
        <Button variant="ghost" onClick={() => navigate(`/tools/${namespace}`)}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
        <p className="mt-4 text-muted-foreground">Tool "{toolName}" not found.</p>
      </div>
    )
  }

  return (
    <div className="mx-auto w-full max-w-2xl space-y-4">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" onClick={() => navigate(`/tools/${namespace}`)}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <span
          className="h-3 w-3 rounded-full"
          style={{ backgroundColor: nsInfo?.color }}
        />
        <span className="text-sm text-muted-foreground">{nsInfo?.label}</span>
        <span className="text-muted-foreground">/</span>
        <h2 className="font-semibold">{toolShortName}</h2>
      </div>

      {tool.description && (
        <p className="text-sm text-muted-foreground">{tool.description}</p>
      )}
      {supportsMultimodalMessages && (
        <p className="text-xs text-muted-foreground">
          This tool accepts structured `messages` blocks. For image inputs, upload files first and reference the returned `image_path` values.
        </p>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            Parameters
            {tool.inputSchema?.required && (
              <Badge variant="outline" className="text-[10px]">
                {tool.inputSchema.required.length} required
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ToolParameterForm
            key={toolName}
            toolName={toolName}
            schema={tool.inputSchema ?? { properties: {} }}
            isLoading={execution.isPending}
            onSubmit={(args) => {
              startTimeRef.current = performance.now()
              execution.mutate({ toolName, args })
            }}
          />
        </CardContent>
      </Card>

      {execution.data && (
        <ToolResultPanel
          toolName={toolName}
          result={execution.data}
          executionTime={performance.now() - startTimeRef.current}
        />
      )}

      {execution.error && (
        <Card className="border-destructive">
          <CardContent className="p-4">
            <p className="text-sm text-destructive">{execution.error.message}</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
