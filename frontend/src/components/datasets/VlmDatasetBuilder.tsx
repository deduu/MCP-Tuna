import { useRef, useState, type ChangeEvent } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { FolderOpen, ImagePlus, Loader2, Save } from 'lucide-react'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { buildDatasetOutputPath } from '@/lib/dataset-output'
import { uploadAsset } from '@/lib/uploads'
import { toast } from 'sonner'

interface UploadedImageAsset {
  image_path: string
  preview_url?: string
  file_name: string
}

function buildVlmRow(
  images: UploadedImageAsset[],
  userPrompt: string,
  assistantResponse: string,
  systemPrompt: string,
) {
  const userContent = [
    ...images.map((image) => ({ type: 'image_path', image_path: image.image_path })),
    ...(userPrompt.trim() ? [{ type: 'text', text: userPrompt.trim() }] : []),
  ]

  const messages: Array<Record<string, unknown>> = []
  if (systemPrompt.trim()) {
    messages.push({
      role: 'system',
      content: [{ type: 'text', text: systemPrompt.trim() }],
    })
  }
  messages.push({ role: 'user', content: userContent })
  messages.push({
    role: 'assistant',
    content: [{ type: 'text', text: assistantResponse.trim() }],
  })

  return { messages }
}

export function VlmDatasetBuilder() {
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const [images, setImages] = useState<UploadedImageAsset[]>([])
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful vision-language assistant.')
  const [userPrompt, setUserPrompt] = useState('')
  const [assistantResponse, setAssistantResponse] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [saveResult, setSaveResult] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  async function handlePickImages(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.currentTarget.files ?? [])
    if (!files.length) return

    setIsUploading(true)
    try {
      const uploaded: UploadedImageAsset[] = []
      for (const file of files) {
        const result = await uploadAsset(file, 'images')
        uploaded.push({
          image_path: result.filePath,
          preview_url: result.previewUrl,
          file_name: result.fileName,
        })
      }
      setImages((current) => [...current, ...uploaded])
      if (!datasetPath.trim()) {
        setDatasetPath(buildDatasetOutputPath(uploaded[0].image_path, 'vlm_sft'))
      }
      toast.success(`Uploaded ${uploaded.length} image${uploaded.length === 1 ? '' : 's'}`)
    } catch (error) {
      toast.error(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsUploading(false)
      event.currentTarget.value = ''
    }
  }

  function removeImage(index: number) {
    setImages((current) => {
      const next = [...current]
      const removed = next.splice(index, 1)[0]
      if (removed?.preview_url) URL.revokeObjectURL(removed.preview_url)
      return next
    })
  }

  async function handleSave() {
    if (images.length === 0) {
      toast.error('At least one image is required')
      return
    }
    if (!assistantResponse.trim()) {
      toast.error('Assistant response is required')
      return
    }
    if (!datasetPath.trim()) {
      toast.error('Dataset output path is required')
      return
    }

    const newRow = buildVlmRow(images, userPrompt, assistantResponse, systemPrompt)
    let dataPoints: Array<Record<string, unknown>> = []

    try {
      const loaded = await executeTool({
        toolName: 'dataset.load',
        args: { file_path: datasetPath.trim() },
      })
      const payload = loaded as Record<string, unknown>
      dataPoints = Array.isArray(payload.data_points)
        ? (payload.data_points as Array<Record<string, unknown>>)
        : []
    } catch {
      dataPoints = []
    }

    dataPoints.push(newRow)

    try {
      const saved = await executeTool({
        toolName: 'dataset.save',
        args: {
          data_points: dataPoints,
          output_path: datasetPath.trim(),
          format: 'jsonl',
        },
      })
      setSaveResult(JSON.stringify(saved, null, 2))
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success(`Saved ${dataPoints.length} VLM row${dataPoints.length === 1 ? '' : 's'} to ${datasetPath.trim()}`)
      setUserPrompt('')
      setAssistantResponse('')
    } catch (error) {
      toast.error(`Save failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <ImagePlus className="h-4 w-4" />
          Build VLM Dataset Row
        </CardTitle>
        <CardDescription>
          Create canonical `vlm_sft` message rows from uploaded images, prompt text, and an assistant answer.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Images</label>
          <div className="flex flex-wrap gap-2">
            {images.map((image, index) => (
              <div key={`${image.image_path}-${index}`} className="relative overflow-hidden rounded-lg border border-border/70 bg-secondary/20">
                {image.preview_url ? (
                  <img src={image.preview_url} alt={image.file_name} className="h-24 w-24 object-cover" />
                ) : (
                  <div className="flex h-24 w-24 items-center justify-center px-2 text-[11px] text-muted-foreground">
                    {image.file_name}
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => removeImage(index)}
                  className="absolute right-1 top-1 rounded-full bg-background/90 px-1.5 py-0.5 text-[10px] text-muted-foreground hover:text-foreground"
                >
                  Remove
                </button>
              </div>
            ))}
            <Button
              type="button"
              variant="outline"
              onClick={() => inputRef.current?.click()}
              disabled={isUploading}
              className="h-24 min-w-24"
            >
              {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <FolderOpen className="h-4 w-4" />}
              {isUploading ? 'Uploading...' : 'Browse'}
            </Button>
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              accept="image/*"
              multiple
              onChange={handlePickImages}
              disabled={isUploading}
            />
          </div>
          {images.length > 0 && (
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline">{images.length} image{images.length === 1 ? '' : 's'}</Badge>
              <Badge variant="outline">Technique: VLM SFT</Badge>
            </div>
          )}
        </div>

        <div className="space-y-1">
          <label className="text-sm font-medium text-foreground">System Prompt</label>
          <Input value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} />
        </div>

        <div className="space-y-1">
          <label className="text-sm font-medium text-foreground">User Prompt</label>
          <textarea
            value={userPrompt}
            onChange={(event) => setUserPrompt(event.target.value)}
            rows={3}
            placeholder="Describe the defect in this image."
            className="min-h-[84px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
        </div>

        <div className="space-y-1">
          <label className="text-sm font-medium text-foreground">Assistant Response</label>
          <textarea
            value={assistantResponse}
            onChange={(event) => setAssistantResponse(event.target.value)}
            rows={4}
            placeholder="The image shows..."
            className="min-h-[96px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
        </div>

        <div className="space-y-1">
          <label className="text-sm font-medium text-foreground">Dataset Output Path</label>
          <Input
            value={datasetPath}
            onChange={(event) => setDatasetPath(event.target.value)}
            placeholder="data/example_vlm_sft.jsonl"
          />
          <p className="text-xs text-muted-foreground">
            Saving appends this row to an existing dataset file if it already exists.
          </p>
        </div>

        <Button onClick={handleSave} disabled={isPending || isUploading}>
          <Save className="h-4 w-4" />
          Save VLM Row
        </Button>

        {saveResult && (
          <pre className="max-h-56 overflow-auto rounded-md bg-secondary/50 p-3 text-xs">
            {saveResult}
          </pre>
        )}
      </CardContent>
    </Card>
  )
}
