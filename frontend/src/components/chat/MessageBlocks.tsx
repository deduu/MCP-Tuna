import type { ChatContentBlock } from '@/lib/chat-content'

interface MessageBlocksProps {
  blocks: ChatContentBlock[]
  className?: string
}

export function MessageBlocks({ blocks, className }: MessageBlocksProps) {
  const textBlocks = blocks.filter((block): block is Extract<ChatContentBlock, { type: 'text' }> => block.type === 'text')
  const imageBlocks = blocks.filter((block): block is Extract<ChatContentBlock, { type: 'image_path' }> => block.type === 'image_path')

  return (
    <div className={className ?? 'space-y-3'}>
      {imageBlocks.length > 0 && (
        <div className="flex flex-wrap gap-3">
          {imageBlocks.map((block) => (
            <figure
              key={`${block.image_path}-${block.file_name ?? ''}`}
              className="overflow-hidden rounded-lg border border-border/70 bg-secondary/20"
            >
              {block.preview_url ? (
                <img
                  src={block.preview_url}
                  alt={block.file_name ?? 'Uploaded image'}
                  className="h-32 w-32 object-cover"
                />
              ) : (
                <div className="flex h-32 w-32 items-center justify-center px-3 text-center text-xs text-muted-foreground">
                  {block.file_name ?? block.image_path.split(/[\\/]/).pop() ?? 'Image'}
                </div>
              )}
              <figcaption className="max-w-32 truncate border-t border-border/60 px-2 py-1 text-[11px] text-muted-foreground">
                {block.file_name ?? block.image_path.split(/[\\/]/).pop() ?? block.image_path}
              </figcaption>
            </figure>
          ))}
        </div>
      )}
      {textBlocks.length > 0 && (
        <div className="space-y-2">
          {textBlocks.map((block, index) => (
            <p key={`${block.text}-${index}`} className="whitespace-pre-wrap text-sm leading-relaxed">
              {block.text}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}
