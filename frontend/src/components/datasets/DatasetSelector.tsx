import { useDatasets } from '@/api/hooks/useDatasets'

interface DatasetSelectorProps {
  value: string
  onChange: (path: string) => void
  label?: string
}

export function DatasetSelector({ value, onChange, label }: DatasetSelectorProps) {
  const { data: datasets } = useDatasets()

  function getDisplayName(filePath: string, format: string, rowCount: number): string {
    const name = filePath.split(/[\\/]/).pop() ?? filePath
    return `${name} (${format}, ${rowCount.toLocaleString()} rows)`
  }

  return (
    <div className="space-y-1.5">
      {label && (
        <label className="text-sm font-medium">{label}</label>
      )}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full h-9 rounded-md border border-input bg-transparent px-3 text-sm text-foreground"
      >
        <option value="">
          {datasets?.length ? 'Select a dataset...' : 'No datasets available'}
        </option>
        {datasets?.map((d) => (
          <option key={d.dataset_id} value={d.file_path}>
            {getDisplayName(d.file_path, d.format, d.row_count)}
          </option>
        ))}
      </select>
    </div>
  )
}
