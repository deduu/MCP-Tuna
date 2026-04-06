import type { DatasetInfo } from '@/api/types'
import { BrowsePathField } from '@/components/evaluation/BrowsePathField'
import { Badge } from '@/components/ui/badge'

interface TrainingDatasetFieldProps {
  label?: string
  datasetPath: string
  onChange: (value: string) => void
  datasets: DatasetInfo[]
  schemaValid: 'pass' | 'warn' | null
  qualityValid: 'pass' | 'warn' | null
  placeholder: string
  hint?: string | null
}

export function TrainingDatasetField({
  label = 'Dataset',
  datasetPath,
  onChange,
  datasets,
  schemaValid,
  qualityValid,
  placeholder,
  hint,
}: TrainingDatasetFieldProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium">{label}</label>
        {schemaValid && (
          <Badge variant={schemaValid === 'pass' ? 'success' : 'warning'} className="text-[10px] px-1.5 py-0">
            schema {schemaValid}
          </Badge>
        )}
        {qualityValid && (
          <Badge variant={qualityValid === 'pass' ? 'success' : 'warning'} className="text-[10px] px-1.5 py-0">
            quality {qualityValid}
          </Badge>
        )}
      </div>
      <BrowsePathField
        value={datasetPath}
        onChange={onChange}
        placeholder={placeholder}
        allowFiles
        allowDirectories={false}
        preferredRootIds={['workspace', 'uploads', 'output']}
      />
      {datasets.length > 0 && (
        <select
          className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          value=""
          onChange={(event) => {
            if (event.target.value) onChange(event.target.value)
          }}
        >
          <option value="">Pick an existing dataset...</option>
          {datasets.map((dataset) => (
            <option key={dataset.file_path} value={dataset.file_path}>
              {dataset.file_path} ({dataset.row_count} rows)
            </option>
          ))}
        </select>
      )}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}
