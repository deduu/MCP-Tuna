import { useEffect, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { ImagePlus, Send, Trash2, X } from 'lucide-react'
import type { Deployment } from '@/api/types'
import { mcpCall } from '@/api/client'
import type { ChatContentBlock, ChatImageBlock } from '@/lib/chat-content'
import { buildUserChatContent, sanitizeChatContentForRequest } from '@/lib/chat-content'
import { uploadAsset } from '@/lib/uploads'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { MessageBlocks } from '@/components/chat/MessageBlocks'
import { toast } from 'sonner'

type DeploymentChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  parts?: ChatContentBlock[]
  error?: boolean
}

type HostChatResult = {
  success: boolean
  conversation_id: string
  deployment_id?: string | null
  response: string
  turns: number
}

interface DeploymentChatProps {
  deployment: Deployment
}

export function DeploymentChat({ deployment }: DeploymentChatProps) {
  const [messages, setMessages] = useState<DeploymentChatMessage[]>([])
  const [input, setInput] = useState('')
  const [imageBlocks, setImageBlocks] = useState<ChatImageBlock[]>([])
  const [isUploadingImage, setIsUploadingImage] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  const chatMutation = useMutation<HostChatResult, Error, string>({
    onMutate: (message) => {
      const content = buildUserChatContent(message, imageBlocks)
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'user',
          content: message,
          parts: deployment.modality === 'vision-language' && Array.isArray(content) ? content : undefined,
        },
      ])
    },
    mutationFn: async (message) => {
      if (deployment.modality === 'vision-language') {
        return await mcpCall<HostChatResult>('host.chat_vlm', {
          deployment_id: deployment.deployment_id,
          messages: [
            {
              role: 'user',
              content: sanitizeChatContentForRequest(buildUserChatContent(message, imageBlocks)),
            },
          ],
          ...(conversationId ? { conversation_id: conversationId } : {}),
        })
      }
      return await mcpCall<HostChatResult>('host.chat', {
        deployment_id: deployment.deployment_id,
        message,
        ...(conversationId ? { conversation_id: conversationId } : {}),
      })
    },
    onSuccess: (result) => {
      setConversationId(result.conversation_id)
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'assistant', content: result.response },
      ])
      for (const block of imageBlocks) {
        if (block.preview_url) URL.revokeObjectURL(block.preview_url)
      }
      setImageBlocks([])
    },
    onError: (error) => {
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'assistant', content: error.message, error: true },
      ])
    },
  })

  useEffect(() => {
    const el = scrollRef.current
    if (!el) {
      return
    }
    el.scrollTop = el.scrollHeight
  }, [messages, chatMutation.isPending])

  useEffect(() => {
    setMessages([])
    setInput('')
    setConversationId(null)
    chatMutation.reset()
  }, [deployment.deployment_id])

  const handleSubmit = () => {
    const trimmed = input.trim()
    if ((!trimmed && imageBlocks.length === 0) || chatMutation.isPending || isUploadingImage || deployment.status !== 'running') {
      return
    }
    setInput('')
    chatMutation.mutate(trimmed)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSubmit()
    }
  }

  const handleClear = () => {
    if (chatMutation.isPending) {
      return
    }
    for (const block of imageBlocks) {
      if (block.preview_url) URL.revokeObjectURL(block.preview_url)
    }
    setMessages([])
    setInput('')
    setImageBlocks([])
    setConversationId(null)
    chatMutation.reset()
  }

  const subtitle =
    deployment.type === 'api'
      ? deployment.modality === 'vision-language'
        ? 'Messages are sent through the deployed VLM API runtime.'
        : 'Messages are sent through the deployed API runtime.'
      : deployment.modality === 'vision-language'
        ? 'Messages use the live deployed VLM runtime managed by the gateway.'
        : 'Messages use the live deployed model runtime managed by the gateway.'

  async function handlePickImage(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.currentTarget.files ?? [])
    if (!files.length) return

    setIsUploadingImage(true)
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
      setImageBlocks((current) => [...current, ...uploadedBlocks])
      toast.success(`Uploaded ${uploadedBlocks.length} image${uploadedBlocks.length === 1 ? '' : 's'}`)
    } catch (error) {
      toast.error(`Image upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsUploadingImage(false)
      event.currentTarget.value = ''
    }
  }

  function removeImageBlock(index: number) {
    setImageBlocks((current) => {
      const next = [...current]
      const removed = next.splice(index, 1)[0]
      if (removed?.preview_url) URL.revokeObjectURL(removed.preview_url)
      return next
    })
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-lg">Deployment Chat</CardTitle>
            <CardDescription>{subtitle}</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {conversationId && (
              <Badge variant="outline" className="font-mono text-[10px]">
                {conversationId}
              </Badge>
            )}
            <Button variant="ghost" size="sm" onClick={handleClear} disabled={chatMutation.isPending}>
              <Trash2 className="h-4 w-4" />
              Clear
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div
          ref={scrollRef}
          className="min-h-[280px] max-h-[420px] overflow-y-auto rounded-lg border bg-secondary/20 p-4"
        >
          {messages.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Send a prompt to verify the deployed model can answer interactively.
            </p>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => (
                <div key={message.id} className="space-y-1">
                  <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                    {message.role === 'user' ? 'User' : message.error ? 'Error' : 'Assistant'}
                  </div>
                  {message.parts ? (
                    <MessageBlocks blocks={message.parts} />
                  ) : (
                    <div className={message.error ? 'text-sm text-destructive whitespace-pre-wrap' : 'text-sm whitespace-pre-wrap'}>
                      {message.content}
                    </div>
                  )}
                </div>
              ))}
              {chatMutation.isPending && (
                <div className="space-y-1">
                  <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                    Assistant
                  </div>
                  <div className="text-sm text-muted-foreground">Generating response...</div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="space-y-2">
          {deployment.modality === 'vision-language' && imageBlocks.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {imageBlocks.map((block, index) => (
                <div key={`${block.image_path}-${index}`} className="relative overflow-hidden rounded-lg border border-border/70 bg-secondary/20">
                  {block.preview_url ? (
                    <img src={block.preview_url} alt={block.file_name ?? 'Uploaded image'} className="h-20 w-20 object-cover" />
                  ) : (
                    <div className="flex h-20 w-20 items-center justify-center px-2 text-[11px] text-muted-foreground">
                      {block.file_name ?? 'Image'}
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={() => removeImageBlock(index)}
                    className="absolute right-1 top-1 rounded-full bg-background/90 p-1 text-muted-foreground hover:text-foreground"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
            disabled={chatMutation.isPending || isUploadingImage || deployment.status !== 'running'}
            placeholder={
              deployment.status === 'running'
                ? deployment.modality === 'vision-language'
                  ? 'Ask the deployed vision-language model a question or attach images...'
                  : 'Ask the deployed model a question...'
                : 'Start or redeploy the model to chat with it.'
            }
            className="flex min-h-[96px] w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
          />
          <div className="flex items-center justify-between gap-3">
            <p className="text-[11px] text-muted-foreground">
              {deployment.modality === 'vision-language'
                ? 'Enter to send, Shift+Enter for a new line, image button to attach'
                : 'Enter to send, Shift+Enter for a new line'}
            </p>
            <div className="flex items-center gap-2">
              {deployment.modality === 'vision-language' && (
                <>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => imageInputRef.current?.click()}
                    disabled={chatMutation.isPending || isUploadingImage || deployment.status !== 'running'}
                  >
                    <ImagePlus className="h-4 w-4" />
                    Attach image
                  </Button>
                  <input
                    ref={imageInputRef}
                    type="file"
                    className="hidden"
                    accept="image/*"
                    multiple
                    onChange={handlePickImage}
                    disabled={chatMutation.isPending || isUploadingImage}
                  />
                </>
              )}
              <Button onClick={handleSubmit} disabled={(!input.trim() && imageBlocks.length === 0) || chatMutation.isPending || isUploadingImage || deployment.status !== 'running'}>
                <Send className="h-4 w-4" />
                {chatMutation.isPending ? 'Sending...' : 'Send'}
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
