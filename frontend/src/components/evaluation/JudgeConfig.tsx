import { RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useJudgeConfig } from '@/api/hooks/useEvaluation'

export function JudgeConfig() {
  const { data, isLoading, refetch, isRefetching } = useJudgeConfig()

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading judge capabilities...</p>
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isRefetching}>
          <RefreshCw className="h-3.5 w-3.5" />
          {isRefetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      </div>
      <pre className="max-h-56 overflow-auto rounded-md bg-secondary/40 p-3 text-xs font-mono">
        {JSON.stringify(data ?? {}, null, 2)}
      </pre>
    </div>
  )
}
