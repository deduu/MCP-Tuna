import { useEffect, useState } from 'react'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

interface DatasetPreviewProps {
  filePath: string
}

interface PreviewData {
  success?: boolean
  error?: string
  rows: Record<string, unknown>[]
  total_rows?: number
}

export function DatasetPreview({ filePath }: DatasetPreviewProps) {
  const { mutateAsync: executeTool } = useToolExecution()
  const [data, setData] = useState<PreviewData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function load() {
      setLoading(true)
      setError(null)
      try {
        const result = await executeTool({
          toolName: 'dataset.preview',
          args: { file_path: filePath, n: 50 },
        })
        const payload = result as Record<string, unknown>
        if (payload.success === false) {
          const message =
            typeof payload.error === 'string' && payload.error.trim()
              ? payload.error
              : 'Failed to load preview'
          throw new Error(message)
        }
        if (!cancelled) {
          setData(payload as unknown as PreviewData)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load preview')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [filePath, executeTool])

  if (loading) {
    return (
      <div className="space-y-2 pt-2">
        <Skeleton className="h-6 w-full" />
        <Skeleton className="h-6 w-full" />
        <Skeleton className="h-6 w-full" />
      </div>
    )
  }

  if (error) {
    return <p className="text-xs text-destructive pt-2">{error}</p>
  }

  if (!data?.rows?.length) {
    return <p className="text-xs text-muted-foreground pt-2">No rows to display</p>
  }

  const columns = Object.keys(data.rows[0])

  function truncate(value: unknown): string {
    const str = typeof value === 'string' ? value : JSON.stringify(value) ?? ''
    return str.length > 100 ? str.slice(0, 100) + '...' : str
  }

  return (
    <div className="pt-2 space-y-2">
      <Badge variant="secondary" className="text-[10px]">
        {data.rows.length} rows{data.total_rows ? ` of ${data.total_rows}` : ''}
      </Badge>
      <div className="overflow-auto max-h-64 rounded border border-border">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border bg-secondary/50">
              {columns.map((col) => (
                <th key={col} className="px-2 py-1.5 text-left font-medium text-muted-foreground whitespace-nowrap">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row, i) => (
              <tr key={i} className="border-b border-border last:border-0 hover:bg-secondary/30">
                {columns.map((col) => (
                  <td key={col} className="px-2 py-1 whitespace-nowrap max-w-[200px] truncate">
                    {truncate(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
