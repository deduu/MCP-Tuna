import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Check, Copy, ChevronDown, ChevronRight } from 'lucide-react'
import type { MCPToolResult } from '@/api/types'

interface ToolResultPanelProps {
  result: MCPToolResult
  executionTime?: number
}

function JsonNode({ data, depth = 0 }: { data: unknown; depth?: number }) {
  const [collapsed, setCollapsed] = useState(depth > 2)

  if (data === null) return <span className="text-muted-foreground">null</span>
  if (data === undefined) return <span className="text-muted-foreground">undefined</span>
  if (typeof data === 'boolean') return <span className="text-amber-400">{String(data)}</span>
  if (typeof data === 'number') return <span className="text-blue-400">{data}</span>
  if (typeof data === 'string') {
    if (data.length > 200) {
      return <span className="text-emerald-400">"{data.slice(0, 200)}..."</span>
    }
    return <span className="text-emerald-400">"{data}"</span>
  }

  if (Array.isArray(data)) {
    if (data.length === 0) return <span className="text-muted-foreground">[]</span>
    return (
      <div>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="inline-flex items-center gap-0.5 text-muted-foreground hover:text-foreground"
        >
          {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          <span className="text-xs">[{data.length}]</span>
        </button>
        {!collapsed && (
          <div className="ml-4 border-l border-border pl-2 space-y-0.5">
            {data.map((item, i) => (
              <div key={i}>
                <span className="text-muted-foreground text-xs mr-1">{i}:</span>
                <JsonNode data={item} depth={depth + 1} />
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  if (typeof data === 'object') {
    const entries = Object.entries(data as Record<string, unknown>)
    if (entries.length === 0) return <span className="text-muted-foreground">{'{}'}</span>
    return (
      <div>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="inline-flex items-center gap-0.5 text-muted-foreground hover:text-foreground"
        >
          {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          <span className="text-xs">{`{${entries.length}}`}</span>
        </button>
        {!collapsed && (
          <div className="ml-4 border-l border-border pl-2 space-y-0.5">
            {entries.map(([key, val]) => (
              <div key={key}>
                <span className="text-purple-400 text-xs">{key}</span>
                <span className="text-muted-foreground text-xs">: </span>
                <JsonNode data={val} depth={depth + 1} />
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  return <span>{String(data)}</span>
}

export function ToolResultPanel({ result, executionTime }: ToolResultPanelProps) {
  const [copied, setCopied] = useState(false)
  const isError = result.success === false || (typeof result.error === 'string' && result.error.trim().length > 0)
  const statusLabel = isError ? 'Error' : 'Success'
  const statusVariant = isError ? 'error' : 'success'

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between py-3">
        <div className="flex items-center gap-2">
          <CardTitle className="text-sm">Result</CardTitle>
          <Badge variant={statusVariant}>
            {statusLabel}
          </Badge>
          {executionTime !== undefined && (
            <span className="text-xs text-muted-foreground font-mono">
              {executionTime.toFixed(0)}ms
            </span>
          )}
        </div>
        <Button variant="ghost" size="sm" onClick={handleCopy} className="gap-1">
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </CardHeader>
      <CardContent className="font-mono text-xs overflow-auto max-h-96">
        <JsonNode data={result} />
      </CardContent>
    </Card>
  )
}
