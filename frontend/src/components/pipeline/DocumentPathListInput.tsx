import { type ChangeEvent, useEffect, useRef, useState } from 'react'
import { FolderOpen, Files, Loader2 } from 'lucide-react'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'

const DOCUMENT_FILE_ACCEPT = '.pdf,.md,.markdown,.txt,.doc,.docx,.json,.jsonl,.csv,.parquet'

interface DocumentPathListInputProps {
  value: string
  onChange: (value: string) => void
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

function sanitizeSegment(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'upload'
}

function mergePathLists(current: string, incoming: string[]): string {
  const existing = current
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean)
  const merged = [...existing]
  for (const item of incoming) {
    if (!merged.includes(item)) merged.push(item)
  }
  return merged.join('\n')
}

export function DocumentPathListInput({
  value,
  onChange,
  disabled = false,
  helperText = 'Browse Files uploads selected documents. Browse Folder uploads all supported files from one folder.',
}: DocumentPathListInputProps) {
  const filesInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)
  const { mutateAsync: executeTool } = useToolExecution()
  const [isUploading, setIsUploading] = useState(false)

  useEffect(() => {
    if (folderInputRef.current) {
      folderInputRef.current.setAttribute('webkitdirectory', '')
      folderInputRef.current.setAttribute('directory', '')
    }
  }, [])

  async function uploadDocument(file: File): Promise<string> {
    const relativePath =
      'webkitRelativePath' in file && typeof file.webkitRelativePath === 'string' && file.webkitRelativePath
        ? file.webkitRelativePath
        : file.name
    const segments = relativePath
      .split(/[\\/]/)
      .filter(Boolean)
      .map(sanitizeSegment)
    const serverFilename = `documents/${crypto.randomUUID()}_${segments.join('/')}`
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

  async function handlePickMany(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.currentTarget.files ?? [])
    if (!files.length) return

    setIsUploading(true)
    try {
      const uploadedPaths: string[] = []
      for (const file of files) {
        uploadedPaths.push(await uploadDocument(file))
      }
      onChange(mergePathLists(value, uploadedPaths))
      toast.success(`Uploaded ${uploadedPaths.length} document${uploadedPaths.length === 1 ? '' : 's'}`)
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
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={4}
        placeholder={'/path/to/doc1.pdf\n/path/to/doc2.md'}
        className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
        disabled={isDisabled}
      />
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant="outline"
          onClick={() => filesInputRef.current?.click()}
          disabled={isDisabled}
        >
          {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Files className="h-4 w-4" />}
          Browse Files
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={() => folderInputRef.current?.click()}
          disabled={isDisabled}
        >
          {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <FolderOpen className="h-4 w-4" />}
          Browse Folder
        </Button>
      </div>
      <input
        ref={filesInputRef}
        type="file"
        className="hidden"
        accept={DOCUMENT_FILE_ACCEPT}
        multiple
        onChange={handlePickMany}
        disabled={isDisabled}
      />
      <input
        ref={folderInputRef}
        type="file"
        className="hidden"
        accept={DOCUMENT_FILE_ACCEPT}
        multiple
        onChange={handlePickMany}
        disabled={isDisabled}
      />
      <p className="text-xs text-muted-foreground">{helperText}</p>
    </div>
  )
}
