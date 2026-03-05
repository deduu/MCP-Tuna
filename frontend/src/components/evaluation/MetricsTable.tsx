import { Skeleton } from '@/components/ui/skeleton'

interface MetricsTableProps {
  data: Record<string, unknown> | null
  isLoading?: boolean
}

function formatValue(value: unknown): string {
  if (value == null) return '-'
  if (typeof value === 'number') return value.toFixed(4)
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function isNestedObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasNestedValues(data: Record<string, unknown>): boolean {
  return Object.values(data).some(isNestedObject)
}

export function MetricsTable({ data, isLoading }: MetricsTableProps) {
  if (isLoading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-8 w-full" />
        ))}
      </div>
    )
  }

  if (!data || Object.keys(data).length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-4 text-center">
        No metrics available
      </p>
    )
  }

  // Nested: per-metric objects with sub-scores
  if (hasNestedValues(data)) {
    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-2 pr-4 font-medium text-muted-foreground">Metric</th>
              <th className="text-left py-2 pr-4 font-medium text-muted-foreground">Details</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data).map(([key, value]) => (
              <tr key={key} className="border-b border-border/50">
                <td className="py-2 pr-4 font-medium align-top">{key}</td>
                <td className="py-2">
                  {isNestedObject(value) ? (
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      {Object.entries(value).map(([subKey, subVal]) => (
                        <div key={subKey} className="flex gap-1">
                          <span className="text-muted-foreground">{subKey}:</span>
                          <span>{formatValue(subVal)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <span>{formatValue(value)}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  // Flat key-value
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-2 pr-4 font-medium text-muted-foreground">Metric</th>
            <th className="text-left py-2 font-medium text-muted-foreground">Value</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data).map(([key, value]) => (
            <tr key={key} className="border-b border-border/50">
              <td className="py-2 pr-4 font-medium">{key}</td>
              <td className="py-2">{formatValue(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
