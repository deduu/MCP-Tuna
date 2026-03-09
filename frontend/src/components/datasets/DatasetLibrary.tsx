import { useMemo, useState } from 'react'
import type { DatasetInfo } from '@/api/types'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Search, Upload } from 'lucide-react'
import { DatasetCard } from './DatasetCard'

interface DatasetLibraryProps {
  datasets: DatasetInfo[]
  isLoading: boolean
  onSwitchToImport: () => void
}

export function DatasetLibrary({ datasets, isLoading, onSwitchToImport }: DatasetLibraryProps) {
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'size'>('name')

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
            <DatasetCard key={dataset.file_path} dataset={dataset} />
          ))}
        </div>
      )}
    </div>
  )
}
