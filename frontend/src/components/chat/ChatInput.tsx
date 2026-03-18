import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import { ImagePlus, Send, Square, X } from 'lucide-react'
import { sendChatMessage } from '@/api/chat-client'
import { useDeployments } from '@/api/hooks/useDeployments'
import type { ChatImageBlock } from '@/lib/chat-content'
import { buildUserChatContent } from '@/lib/chat-content'
import { uploadAsset } from '@/lib/uploads'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useChatStore } from '@/stores/chat'
import { toast } from 'sonner'

const AVAILABLE_MODELS = [
  { id: 'gpt-4o', label: 'GPT-4o' },
  { id: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4' },
  { id: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
]

export function ChatInput() {
  const [input, setInput] = useState('')
  const [imageBlocks, setImageBlocks] = useState<ChatImageBlock[]>([])
  const [isUploadingImage, setIsUploadingImage] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  const chatMode = useChatStore((state) => state.chatMode)
  const selectedModel = useChatStore((state) => state.selectedModel)
  const selectedDeploymentId = useChatStore((state) => state.selectedDeploymentId)
  const setChatMode = useChatStore((state) => state.setChatMode)
  const setSelectedModel = useChatStore((state) => state.setSelectedModel)
  const setSelectedDeploymentId = useChatStore((state) => state.setSelectedDeploymentId)
  const isStreaming = useChatStore((state) => state.isStreaming)
  const abortController = useChatStore((state) => state.abortController)

  const { data: deployments } = useDeployments()

  const runningDeployments = useMemo(
    () => (deployments ?? []).filter((deployment) => deployment.status === 'running'),
    [deployments],
  )

  useEffect(() => {
    if (chatMode !== 'deployment') {
      return
    }

    if (runningDeployments.length === 0) {
      if (selectedDeploymentId !== null) {
        setSelectedDeploymentId(null)
      }
      return
    }

    const hasSelectedDeployment = runningDeployments.some(
      (deployment) => deployment.deployment_id === selectedDeploymentId,
    )

    if (!hasSelectedDeployment) {
      setSelectedDeploymentId(runningDeployments[0].deployment_id)
    }
  }, [chatMode, runningDeployments, selectedDeploymentId, setSelectedDeploymentId])

  const selectedDeployment = useMemo(
    () =>
      runningDeployments.find((deployment) => deployment.deployment_id === selectedDeploymentId) ??
      null,
    [runningDeployments, selectedDeploymentId],
  )
  const supportsImages =
    chatMode === 'agent' ||
    selectedDeployment?.modality === 'vision-language'

  const helperText =
    chatMode === 'agent'
      ? supportsImages
        ? 'Tool Agent uses managed providers, can call MCP tools, and accepts image attachments.'
        : 'Tool Agent uses managed providers and can call MCP tools.'
      : selectedDeployment
        ? selectedDeployment.modality === 'vision-language'
          ? `Deployed Local chats directly with ${shortDeploymentLabel(selectedDeployment.model_path)} in multimodal mode. Attach images and text together here.`
          : `Deployed Local chats directly with ${shortDeploymentLabel(selectedDeployment.model_path)}. MCP tools are disabled in this mode.`
        : 'Deployed Local needs a running deployment. Start one from Deployments first.'

  const handleSubmit = useCallback(() => {
    const trimmed = input.trim()
    if ((!trimmed && imageBlocks.length === 0) || isStreaming || isUploadingImage) {
      return
    }

    if (chatMode === 'deployment' && !selectedDeploymentId) {
      return
    }

    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    const payload = buildUserChatContent(trimmed, imageBlocks)
    void sendChatMessage(payload, {
      source: chatMode,
      model: selectedModel,
      deploymentId: selectedDeploymentId,
      deploymentModality: selectedDeployment?.modality === 'vision-language' ? 'vision-language' : 'text',
    })
    for (const block of imageBlocks) {
      if (block.preview_url) {
        URL.revokeObjectURL(block.preview_url)
      }
    }
    setImageBlocks([])
  }, [chatMode, imageBlocks, input, isStreaming, isUploadingImage, selectedDeployment, selectedDeploymentId, selectedModel])

  const handleStop = useCallback(() => {
    abortController?.abort()
  }, [abortController])

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        handleSubmit()
      }
    },
    [handleSubmit],
  )

  const handleInputChange = useCallback((event: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(event.target.value)
    event.target.style.height = 'auto'
    event.target.style.height = `${Math.min(event.target.scrollHeight, 200)}px`
  }, [])

  const handlePickImage = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
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
  }, [])

  const removeImageBlock = useCallback((index: number) => {
    setImageBlocks((current) => {
      const next = [...current]
      const removed = next.splice(index, 1)[0]
      if (removed?.preview_url) {
        URL.revokeObjectURL(removed.preview_url)
      }
      return next
    })
  }, [])

  return (
    <div className="border-t bg-card p-4">
      <div className="mx-auto max-w-3xl space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setChatMode('agent')}
            className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
              chatMode === 'agent'
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400'
                : 'border-border text-muted-foreground hover:bg-accent hover:text-foreground'
            }`}
          >
            Tool Agent
          </button>
          <button
            type="button"
            onClick={() => setChatMode('deployment')}
            className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
              chatMode === 'deployment'
                ? 'border-amber-500/40 bg-amber-500/10 text-amber-400'
                : 'border-border text-muted-foreground hover:bg-accent hover:text-foreground'
            }`}
          >
            Deployed Local
          </button>

          {chatMode === 'agent' ? (
            <>
              <select
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                className="h-9 rounded-md border border-input bg-background px-3 text-xs text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              >
                {AVAILABLE_MODELS.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}
                  </option>
                ))}
              </select>
              <Badge variant="success">MCP tools enabled</Badge>
            </>
          ) : (
            <>
              <select
                value={selectedDeploymentId ?? ''}
                onChange={(event) => setSelectedDeploymentId(event.target.value || null)}
                disabled={runningDeployments.length === 0}
                className="h-9 min-w-[220px] rounded-md border border-input bg-background px-3 text-xs text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              >
                {runningDeployments.length === 0 ? (
                  <option value="">No running deployments</option>
                ) : (
                  runningDeployments.map((deployment) => (
                    <option key={deployment.deployment_id} value={deployment.deployment_id}>
                      {shortDeploymentLabel(deployment.model_path)}
                      {deployment.modality === 'vision-language' ? ' (VLM)' : ''}
                    </option>
                  ))
                )}
              </select>
              <Badge variant="warning">Direct model chat</Badge>
            </>
          )}
        </div>

        {imageBlocks.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {imageBlocks.map((block, index) => (
              <div
                key={`${block.image_path}-${index}`}
                className="relative overflow-hidden rounded-lg border border-border/70 bg-secondary/20"
              >
                {block.preview_url ? (
                  <img
                    src={block.preview_url}
                    alt={block.file_name ?? 'Uploaded image'}
                    className="h-20 w-20 object-cover"
                  />
                ) : (
                  <div className="flex h-20 w-20 items-center justify-center px-2 text-[11px] text-muted-foreground">
                    {block.file_name ?? 'Image'}
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => removeImageBlock(index)}
                  className="absolute right-1 top-1 rounded-full bg-background/90 p-1 text-muted-foreground transition-colors hover:text-foreground"
                  aria-label="Remove image"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-2">
          <div className="relative flex-1">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={isStreaming || isUploadingImage || (chatMode === 'deployment' && !selectedDeploymentId)}
              placeholder={
                chatMode === 'agent'
                  ? supportsImages
                    ? 'Message MCP Tuna or attach images...'
                    : 'Message MCP Tuna...'
                  : selectedDeploymentId
                    ? selectedDeployment?.modality === 'vision-language'
                      ? 'Message the deployed VLM or attach images...'
                      : 'Message the deployed local model...'
                    : 'Select a running deployment first...'
              }
              className="w-full resize-none rounded-lg border border-input bg-background px-4 py-3 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>
          {supportsImages && (
            <>
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={() => imageInputRef.current?.click()}
                disabled={isStreaming || isUploadingImage || (chatMode === 'deployment' && !selectedDeploymentId)}
                className="shrink-0 self-end"
                title="Attach images"
              >
                <ImagePlus className="h-4 w-4" />
              </Button>
              <input
                ref={imageInputRef}
                type="file"
                className="hidden"
                accept="image/*"
                multiple
                onChange={handlePickImage}
                disabled={isStreaming || isUploadingImage}
              />
            </>
          )}
          {isStreaming ? (
            <Button
              variant="destructive"
              size="icon"
              onClick={handleStop}
              className="shrink-0 self-end"
              title="Stop generating"
            >
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              size="icon"
              onClick={handleSubmit}
              disabled={(!input.trim() && imageBlocks.length === 0) || isUploadingImage || (chatMode === 'deployment' && !selectedDeploymentId)}
              className="shrink-0 self-end"
              title="Send message"
            >
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>

        <div className="flex items-center justify-between gap-3">
          <p className="text-[11px] text-muted-foreground">{helperText}</p>
          <p className="text-[10px] text-muted-foreground">
            {supportsImages ? 'Enter to send, Shift+Enter for new line, image button to attach' : 'Enter to send, Shift+Enter for new line'}
          </p>
        </div>
      </div>
    </div>
  )
}

function shortDeploymentLabel(modelPath: string) {
  const normalized = modelPath.replace(/\\/g, '/')
  const parts = normalized.split('/')
  return parts[parts.length - 1] || modelPath
}
