import { useEffect, useMemo, useState } from 'react'
import type { DatasetInfo } from '@/api/types'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Layers2, RefreshCw, Search, Upload, X } from 'lucide-react'
import { toast } from 'sonner'
import {
  formatDatasetLibraryRootsInput,
  parseDatasetLibraryRootsInput,
  resetDatasetLibraryRoots,
  setDatasetLibraryRoots,
} from '@/lib/dataset-library-roots'
import { DatasetCard } from './DatasetCard'
import { SplitMergeDialog } from './SplitMergeDialog'

interface DatasetLibraryProps {
  datasets: DatasetInfo[]
  isLoading: boolean
  scanRoots: string[]
  prunedStaleRecords: number
  onRefresh: () => void
  onScanRootsChange: (roots: string[]) => void
  onSwitchToImport: () => void
}

export function DatasetLibrary({
  datasets,
  isLoading,
  scanRoots,
  prunedStaleRecords,
  onRefresh,
  onScanRootsChange,
  onSwitchToImport,
}: DatasetLibraryProps) {
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'size'>('name')
  const [selectedPaths, setSelectedPaths] = useState<string[]>([])
  const [showMerge, setShowMerge] = useState(false)
  const [rootsDraft, setRootsDraft] = useState(() => formatDatasetLibraryRootsInput(scanRoots))

  useEffect(() => {
    setRootsDraft(formatDatasetLibraryRootsInput(scanRoots))
  }, [scanRoots])

  const filtered = useMemo(() => {
    let result = datasets
    if (search.trim()) {
      const q = search.toLowerCase()
      result = result.filter(
        (d) =>
          d.file_path.toLowerCase().includes(q) ||
          (d.format ?? '').toLowerCase().includes(q),
      )
    }

    return [...result].sort((a, b) => {
      if (sortBy === 'name') {
        const nameA = a.file_path.split(/[\\/]/).pop() ?? ''
        const nameB = b.file_path.split(/[\\/]/).pop() ?? ''
        return nameA.localeCompare(nameB)
      }
      if (sortBy === 'size') return b.size_bytes - a.size_bytes
      const timeA = Date.parse(a.modified_at ?? '')
      const timeB = Date.parse(b.modified_at ?? '')
      if (Number.isFinite(timeA) && Number.isFinite(timeB)) {
        return timeB - timeA
      }
      if (Number.isFinite(timeA)) return -1
      if (Number.isFinite(timeB)) return 1
      return a.file_path.localeCompare(b.file_path)
    })
  }, [datasets, search, sortBy])

  function toggleSelected(filePath: string, selected: boolean) {
    setSelectedPaths((prev) => {
      if (selected) {
        return prev.includes(filePath) ? prev : [...prev, filePath]
      }
      return prev.filter((path) => path !== filePath)
    })
  }

  function handleSaveRoots() {
    const nextRoots = setDatasetLibraryRoots(parseDatasetLibraryRootsInput(rootsDraft))
    setRootsDraft(formatDatasetLibraryRootsInput(nextRoots))
    onScanRootsChange(nextRoots)
    toast.success(`Library now scans ${nextRoots.length} path${nextRoots.length === 1 ? '' : 's'}`)
  }

  function handleResetRoots() {
    const nextRoots = resetDatasetLibraryRoots()
    setRootsDraft(formatDatasetLibraryRootsInput(nextRoots))
    onScanRootsChange(nextRoots)
    toast.success('Library paths reset to the default scan roots')
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-9 w-72" />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48" />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-border/60 bg-secondary/10 p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="space-y-1">
            <h3 className="text-sm font-semibold">Library Paths</h3>
            <p className="text-xs text-muted-foreground">
              One file or directory path per line. Relative paths resolve from the workspace root.
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={onRefresh}>
            <RefreshCw className="h-4 w-4" />
            Refresh
          </Button>
        </div>

        <textarea
          value={rootsDraft}
          onChange={(e) => setRootsDraft(e.target.value)}
          className="min-h-28 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
          placeholder={'data\noutput\nuploads\nnotebooks'}
        />

        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline">
            {scanRoots.length} scan path{scanRoots.length === 1 ? '' : 's'}
          </Badge>
          {prunedStaleRecords > 0 && (
            <Badge variant="warning">
              Removed {prunedStaleRecords} stale record{prunedStaleRecords === 1 ? '' : 's'}
            </Badge>
          )}
          <Button size="sm" onClick={handleSaveRoots}>
            Save Paths
          </Button>
          <Button variant="outline" size="sm" onClick={handleResetRoots}>
            Reset Defaults
          </Button>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search datasets..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>

        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as 'name' | 'date' | 'size')}
          className="h-9 rounded-md border border-input bg-transparent px-3 text-sm text-foreground"
        >
          <option value="name">Sort by Name</option>
          <option value="date">Sort by Date</option>
          <option value="size">Sort by Size</option>
        </select>

        <Button variant="outline" onClick={onSwitchToImport}>
          <Upload className="h-4 w-4" />
          Import Data
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="outline">
          {selectedPaths.length} selected
        </Badge>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowMerge(true)}
          disabled={selectedPaths.length < 2}
        >
          <Layers2 className="h-4 w-4" />
          Merge Selected
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setSelectedPaths([])}
          disabled={selectedPaths.length === 0}
        >
          <X className="h-4 w-4" />
          Clear
        </Button>
      </div>

      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <p className="text-muted-foreground mb-4">No datasets found</p>
          <Button onClick={onSwitchToImport}>
            <Upload className="h-4 w-4" />
            Import your first dataset
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((dataset) => (
            <DatasetCard
              key={dataset.file_path}
              dataset={dataset}
              selected={selectedPaths.includes(dataset.file_path)}
              onToggleSelect={toggleSelected}
            />
          ))}
        </div>
      )}

      <SplitMergeDialog
        open={showMerge}
        onClose={() => setShowMerge(false)}
        mode="merge"
        datasetPaths={selectedPaths}
      />
    </div>
  )
}
