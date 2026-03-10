import { type ChangeEvent, useRef, useState } from 'react'
import { FolderOpen, Loader2 } from 'lucide-react'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { toast } from 'sonner'

const DOCUMENT_FILE_ACCEPT = '.pdf,.md,.markdown,.txt,.doc,.docx,.json,.jsonl,.csv,.parquet'

interface DocumentPathInputProps {
  value: string
  onChange: (path: string) => void
  placeholder?: string
  disabled?: boolean
  helperText?: string
}

async function toBase64(file: File): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result !== 'string') {
        reject(new Error('Failed to read file'))
        return
      }

      const commaIndex = reader.result.indexOf(',')
      resolve(commaIndex >= 0 ? reader.result.slice(commaIndex + 1) : reader.result)
    }
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file'))
    reader.readAsDataURL(file)
  })
}

export function DocumentPathInput({
  value,
  onChange,
  placeholder = '/path/to/document',
  disabled = false,
  helperText = 'Browse uploads the file to the backend and fills in a server path automatically.',
}: DocumentPathInputProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const { mutateAsync: executeTool } = useToolExecution()
  const [isUploading, setIsUploading] = useState(false)

  async function uploadDocument(file: File): Promise<string> {
    const dot = file.name.lastIndexOf('.')
    const baseName = dot >= 0 ? file.name.slice(0, dot) : file.name
    const extension = dot >= 0 ? file.name.slice(dot) : ''
    const safeBaseName = baseName.replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'upload'
    const serverFilename = `documents/${crypto.randomUUID()}_${safeBaseName}${extension}`
    const contentBase64 = await toBase64(file)
    const uploaded = await executeTool({
      toolName: 'file.upload',
      args: { filename: serverFilename, content_base64: contentBase64 },
    })
    const payload = uploaded as Record<string, unknown>
    const uploadedPath = typeof payload.file_path === 'string' ? payload.file_path : ''
    if (!uploadedPath.trim()) {
      throw new Error('Upload succeeded but no server file path was returned')
    }
    return uploadedPath
  }

  async function handlePickFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0]
    if (!file) return

    setIsUploading(true)
    try {
      const uploadedPath = await uploadDocument(file)
      onChange(uploadedPath)
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
