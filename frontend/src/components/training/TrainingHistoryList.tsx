import { ChevronLeft, ChevronRight, Search } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Tabs, Tab, TabList } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import type { TrainingHistoryEntry, TrainingHistoryFilter } from './training-history'
import { getHistoryStatusVariant } from './training-history'

interface TrainingHistoryListProps {
  entries: TrainingHistoryEntry[]
  selectedKey: string | null
  onSelect: (key: string) => void
  filter: TrainingHistoryFilter
  onFilterChange: (filter: TrainingHistoryFilter) => void
  counts: Record<TrainingHistoryFilter, number>
  query: string
  onQueryChange: (value: string) => void
  page: number
  totalPages: number
  totalItems: number
  startIndex: number
  endIndex: number
  onPageChange: (page: number) => void
  isTruncated?: boolean
}

function KindBadge({ kind }: { kind: TrainingHistoryEntry['kind'] }) {
  return (
    <Badge variant={kind === 'pipeline' ? 'secondary' : 'outline'}>
      {kind === 'pipeline' ? 'Pipeline' : 'Direct'}
    </Badge>
  )
}

export function TrainingHistoryList({
  entries,
  selectedKey,
  onSelect,
  filter,
  onFilterChange,
  counts,
  query,
  onQueryChange,
  page,
  totalPages,
  totalItems,
  startIndex,
  endIndex,
  onPageChange,
  isTruncated = false,
}: TrainingHistoryListProps) {
  return (
    <Card className="border-border/70">
      <CardHeader className="space-y-4 pb-4">
        <div>
          <CardTitle className="text-base">Run History</CardTitle>
          <p className="mt-1 text-sm text-muted-foreground">
            Select a run to inspect details without losing your place in the list.
          </p>
        </div>

        <Tabs value={filter} onValueChange={(value) => onFilterChange(value as TrainingHistoryFilter)}>
          <TabList className="mb-0">
            <Tab value="active">Active ({counts.active})</Tab>
            <Tab value="finished">Finished ({counts.finished})</Tab>
            <Tab value="all">All ({counts.all})</Tab>
          </TabList>
        </Tabs>

        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search job id, model, step, or dataset..."
            className="pl-9"
          />
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="space-y-2 xl:max-h-[70vh] xl:overflow-y-auto xl:pr-1">
          {entries.length === 0 ? (
            <div className="rounded-xl border border-dashed border-border/70 px-4 py-8 text-center text-sm text-muted-foreground">
              No runs match the current filters.
            </div>
          ) : (
            entries.map((entry) => (
              <button
                key={entry.key}
                type="button"
                onClick={() => onSelect(entry.key)}
                className={cn(
                  'w-full rounded-xl border p-3 text-left transition-colors',
                  selectedKey === entry.key
                    ? 'border-primary bg-primary/10'
                    : 'border-border/70 bg-secondary/10 hover:border-primary/35 hover:bg-secondary/20',
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 space-y-2">
                    <div className="flex flex-wrap gap-2">
                      <KindBadge kind={entry.kind} />
                      <Badge variant={getHistoryStatusVariant(entry.status)}>{entry.status}</Badge>
                    </div>
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">{entry.title}</p>
                      <p className="truncate text-xs text-muted-foreground">{entry.subtitle}</p>
                    </div>
                  </div>
                  {entry.progress != null && (
                    <span className="text-xs font-mono text-muted-foreground">
                      {Math.round(entry.progress)}%
                    </span>
                  )}
                </div>

                {entry.progress != null && (
                  <Progress value={entry.progress} className="mt-3 h-1.5" />
                )}

                <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                  <span className="truncate">{entry.meta}</span>
                  {entry.updatedLabel && <span>{entry.updatedLabel}</span>}
                </div>
              </button>
            ))
          )}
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-border/70 pt-4">
          <div className="text-xs text-muted-foreground">
            {totalItems > 0 ? `${startIndex + 1}-${endIndex} of ${totalItems}` : '0 results'}
            {isTruncated && ' • showing newest fetched records'}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onPageChange(page - 1)}
              disabled={page <= 1}
              className="gap-1"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
              Prev
            </Button>
            <span className="text-xs text-muted-foreground">
              Page {page} / {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onPageChange(page + 1)}
              disabled={page >= totalPages}
              className="gap-1"
            >
              Next
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
