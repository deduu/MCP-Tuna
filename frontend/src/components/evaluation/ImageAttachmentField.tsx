import { type ChangeEvent } from 'react'
import { ImagePlus, Loader2, X } from 'lucide-react'
import type { ChatImageBlock } from '@/lib/chat-content'
import { uploadAsset } from '@/lib/uploads'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { toast } from 'sonner'

interface ImageAttachmentFieldProps {
  images: ChatImageBlock[]
  onChange: (images: ChatImageBlock[]) => void
  isUploading: boolean
  onUploadingChange: (value: boolean) => void
}

export function ImageAttachmentField({
  images,
  onChange,
  isUploading,
  onUploadingChange,
}: ImageAttachmentFieldProps) {
  async function handlePickImages(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.currentTarget.files ?? [])
    if (!files.length) return

    onUploadingChange(true)
    try {
      const uploadedBlocks: ChatImageBlock[] = []
      for (const file of files) {
        const uploaded = await uploadAsset(file, 'images')
        uploadedBlocks.push({
          type: 'image_path',
          image_path: uploaded.filePath,
          preview_url: uploaded.previewUrl,
          file_name: uploaded.fileName,
        })
      }
      onChange([...images, ...uploadedBlocks])
      toast.success(`Uploaded ${uploadedBlocks.length} image${uploadedBlocks.length === 1 ? '' : 's'}`)
    } catch (error) {
      toast.error(`Image upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      onUploadingChange(false)
      event.currentTarget.value = ''
    }
  }

  function removeImage(index: number) {
    const next = [...images]
    const removed = next.splice(index, 1)[0]
    if (removed?.preview_url) {
      URL.revokeObjectURL(removed.preview_url)
    }
    onChange(next)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <label className="text-sm font-medium">Images (optional)</label>
        {images.length > 0 && <Badge variant="outline">{images.length} attached</Badge>}
      </div>
      <div className="flex flex-wrap gap-2">
        {images.map((image, index) => (
          <div key={`${image.image_path}-${index}`} className="relative overflow-hidden rounded-lg border border-border/70 bg-secondary/20">
            {image.preview_url ? (
              <img src={image.preview_url} alt={image.file_name ?? 'Uploaded image'} className="h-20 w-20 object-cover" />
            ) : (
              <div className="flex h-20 w-20 items-center justify-center px-2 text-[11px] text-muted-foreground">
                {image.file_name ?? 'Image'}
              </div>
            )}
            <button
              type="button"
              onClick={() => removeImage(index)}
              className="absolute right-1 top-1 rounded-full bg-background/90 p-1 text-muted-foreground transition-colors hover:text-foreground"
              aria-label="Remove image"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        ))}
        <label className="inline-flex">
          <input
            type="file"
            className="hidden"
            accept="image/*"
            multiple
            onChange={handlePickImages}
            disabled={isUploading}
          />
          <Button type="button" variant="outline" className="h-20 min-w-20" disabled={isUploading}>
            {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ImagePlus className="h-4 w-4" />}
            {isUploading ? 'Uploading...' : 'Attach'}
          </Button>
        </label>
      </div>
      <p className="text-xs text-muted-foreground">
        Attach one or more images to switch this evaluation to the multimodal judge path.
      </p>
    </div>
  )
}
