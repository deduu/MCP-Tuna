import type { DatasetInfo } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'

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
      <div className="relative">
        <Input
          value={datasetPath}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
        />
        {datasets.length > 0 && (
          <select
            className="absolute right-1 top-1/2 h-7 w-7 -translate-y-1/2 appearance-none rounded border-none bg-transparent text-xs text-muted-foreground opacity-60 cursor-pointer hover:opacity-100"
            value=""
            title="Pick a dataset"
            onChange={(event) => {
              if (event.target.value) onChange(event.target.value)
            }}
          >
            <option value="">▾</option>
            {datasets.map((dataset) => (
              <option key={dataset.file_path} value={dataset.file_path}>
                {dataset.file_path} ({dataset.row_count} rows)
              </option>
            ))}
          </select>
        )}
      </div>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}
