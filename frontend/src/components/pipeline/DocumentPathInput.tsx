import { type ChangeEvent, useRef, useState } from 'react'
import { FolderOpen, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { uploadAsset } from '@/lib/uploads'
import { toast } from 'sonner'

const DOCUMENT_FILE_ACCEPT = '.pdf,.md,.markdown,.txt,.doc,.docx,.json,.jsonl,.csv,.parquet'

interface DocumentPathInputProps {
  value: string
  onChange: (path: string) => void
  placeholder?: string
  disabled?: boolean
  helperText?: string
}

export function DocumentPathInput({
  value,
  onChange,
  placeholder = '/path/to/document',
  disabled = false,
  helperText = 'Browse uploads the file to the backend and fills in a server path automatically.',
}: DocumentPathInputProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isUploading, setIsUploading] = useState(false)

  async function handlePickFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0]
    if (!file) return

    setIsUploading(true)
    try {
      const uploaded = await uploadAsset(file, 'documents')
      onChange(uploaded.filePath)
      toast.success(`Uploaded ${file.name}`)
    } catch (err) {
      toast.error(`Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setIsUploading(false)
      event.currentTarget.value = ''
    }
  }

  const isDisabled = disabled || isUploading

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={isDisabled}
        />
        <Button
          type="button"
          variant="outline"
          onClick={() => inputRef.current?.click()}
          disabled={isDisabled}
        >
          {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <FolderOpen className="h-4 w-4" />}
          {isUploading ? 'Uploading...' : 'Browse'}
        </Button>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={DOCUMENT_FILE_ACCEPT}
          onChange={handlePickFile}
          disabled={isDisabled}
        />
      </div>
      <p className="text-xs text-muted-foreground">{helperText}</p>
    </div>
  )
}
