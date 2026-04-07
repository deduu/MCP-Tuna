import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import {
  Bot,
  ChevronDown,
  ChevronUp,
  ImagePlus,
  Send,
  Server,
  SlidersHorizontal,
  Square,
  Trash2,
  X,
} from 'lucide-react'
import { toast } from 'sonner'
import { sendChatMessage } from '@/api/chat-client'
import { useDeployments } from '@/api/hooks/useDeployments'
import type { ChatImageBlock } from '@/lib/chat-content'
import { buildUserChatContent } from '@/lib/chat-content'
import { deploymentDisplayLabel } from '@/lib/compare-targets'
import { uploadAsset } from '@/lib/uploads'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useChatStore } from '@/stores/chat'
import { AVAILABLE_CHAT_MODELS } from './chat-model-options'

const AGENT_STARTERS = [
  'Summarize the tools I can use for dataset preparation and training.',
  'Help me choose a training recipe for my current dataset and hardware.',
  'Check the system status and tell me what would block a training run.',
]

const DEPLOYMENT_TEXT_STARTERS = [
  'Give me a short summary of what this model is optimized for.',
  'Answer this question step by step and explain your reasoning briefly.',
  'Show me how you handle a concise instruction-following task.',
]

const DEPLOYMENT_VLM_STARTERS = [
  'Describe what stands out in the attached image.',
  'Compare the visual details in the image and summarize the scene.',
  'Extract the key information from the image and explain it clearly.',
]

interface ChatInputProps {
  onClear?: () => void
  clearDisabled?: boolean
}

export function ChatInput({ onClear, clearDisabled = false }: ChatInputProps) {
  const [input, setInput] = useState('')
  const [imageBlocks, setImageBlocks] = useState<ChatImageBlock[]>([])
  const [isUploadingImage, setIsUploadingImage] = useState(false)
  const [deploymentTemperature, setDeploymentTemperature] = useState('0.7')
  const [deploymentMaxNewTokens, setDeploymentMaxNewTokens] = useState('512')
  const [showDeploymentSettings, setShowDeploymentSettings] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  const chatMode = useChatStore((state) => state.chatMode)
  const selectedModel = useChatStore((state) => state.selectedModel)
  const selectedDeploymentId = useChatStore((state) => state.selectedDeploymentId)
  const messageCount = useChatStore((state) => state.messages.length)
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

  useEffect(() => {
    if (chatMode !== 'deployment') {
      setShowDeploymentSettings(false)
    }
  }, [chatMode])

  const selectedDeployment = useMemo(
    () =>
      runningDeployments.find((deployment) => deployment.deployment_id === selectedDeploymentId) ??
      null,
    [runningDeployments, selectedDeploymentId],
  )
  const selectedModelLabel =
    AVAILABLE_CHAT_MODELS.find((model) => model.id === selectedModel)?.label ?? selectedModel
  const supportsImages =
    chatMode === 'agent' || selectedDeployment?.modality === 'vision-language'

  const starterPrompts =
    chatMode === 'agent'
      ? AGENT_STARTERS
      : selectedDeployment?.modality === 'vision-language'
        ? DEPLOYMENT_VLM_STARTERS
        : DEPLOYMENT_TEXT_STARTERS
  const showStarterPrompts =
    messageCount === 0 &&
    input.trim().length === 0 &&
    imageBlocks.length === 0 &&
    !isStreaming
  const selectionSummary =
    chatMode === 'agent'
      ? `${selectedModelLabel} | MCP tools enabled${supportsImages ? ' | images supported' : ''}`
      : selectedDeployment
        ? `${deploymentDisplayLabel(selectedDeployment)}${selectedDeployment.modality === 'vision-language' ? ' | image input supported' : ''}`
        : 'Select a running deployment to start direct chat'
  const showNewChatAction = Boolean(onClear) && messageCount > 0

  const syncTextareaHeight = useCallback((element: HTMLTextAreaElement | null) => {
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`
  }, [])

  const handleSubmit = useCallback(() => {
    const trimmed = input.trim()
    if ((!trimmed && imageBlocks.length === 0) || isStreaming || isUploadingImage) {
      return
    }

    if (chatMode === 'deployment' && !selectedDeploymentId) {
      return
    }

    setInput('')
    syncTextareaHeight(textareaRef.current)

    const payload = buildUserChatContent(trimmed, imageBlocks)
    void sendChatMessage(payload, {
      source: chatMode,
      model: selectedModel,
      temperature: chatMode === 'deployment' ? resolveTemperature(deploymentTemperature) : undefined,
      maxNewTokens: chatMode === 'deployment' ? resolveMaxNewTokens(deploymentMaxNewTokens) : undefined,
      deploymentId: selectedDeploymentId,
      deploymentModality: selectedDeployment?.modality === 'vision-language' ? 'vision-language' : 'text',
    })
    for (const block of imageBlocks) {
      if (block.preview_url) {
        URL.revokeObjectURL(block.preview_url)
      }
    }
    setImageBlocks([])
  }, [
    chatMode,
    deploymentMaxNewTokens,
    deploymentTemperature,
    imageBlocks,
    input,
    isStreaming,
    isUploadingImage,
    selectedDeployment,
    selectedDeploymentId,
    selectedModel,
    syncTextareaHeight,
  ])

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
    syncTextareaHeight(event.target)
  }, [syncTextareaHeight])

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

  const applyStarterPrompt = useCallback((prompt: string) => {
    setInput(prompt)
    requestAnimationFrame(() => {
      syncTextareaHeight(textareaRef.current)
      textareaRef.current?.focus()
    })
  }, [syncTextareaHeight])

  return (
    <div className="border-t border-border/70 bg-background/85 px-4 py-4 backdrop-blur">
      <div className="mx-auto max-w-3xl space-y-4">
        <div className="rounded-full border border-border/80 bg-card/85 p-2 shadow-sm shadow-black/20">
          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex rounded-full bg-muted p-1">
              <button
                type="button"
                onClick={() => setChatMode('agent')}
                className={cn(
                  'inline-flex items-center gap-2 rounded-full px-3 py-2 text-sm transition-colors',
                  chatMode === 'agent'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-foreground',
                )}
              >
                <Bot className="h-4 w-4" />
                Tool Agent
              </button>
              <button
                type="button"
                onClick={() => setChatMode('deployment')}
                className={cn(
                  'inline-flex items-center gap-2 rounded-full px-3 py-2 text-sm transition-colors',
                  chatMode === 'deployment'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-foreground',
                )}
              >
                <Server className="h-4 w-4" />
                Deployed Local
              </button>
            </div>

            <div className="min-w-[220px] flex-1">
              {chatMode === 'agent' ? (
                <select
                  value={selectedModel}
                  onChange={(event) => setSelectedModel(event.target.value)}
                  className="h-10 w-full rounded-full border border-input bg-background px-4 text-sm text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  {AVAILABLE_CHAT_MODELS.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.label}
                    </option>
                  ))}
                </select>
              ) : (
                <select
                  value={selectedDeploymentId ?? ''}
                  onChange={(event) => setSelectedDeploymentId(event.target.value || null)}
                  disabled={runningDeployments.length === 0}
                  className="h-10 w-full rounded-full border border-input bg-background px-4 text-sm text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {runningDeployments.length === 0 ? (
                    <option value="">No running deployments</option>
                  ) : (
                    runningDeployments.map((deployment) => (
                      <option key={deployment.deployment_id} value={deployment.deployment_id}>
                        {deploymentDisplayLabel(deployment)}
                        {deployment.modality === 'vision-language' ? ' (VLM)' : ''}
                      </option>
                    ))
                  )}
                </select>
              )}
            </div>

            {chatMode === 'deployment' && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setShowDeploymentSettings((current) => !current)}
                className="rounded-full"
              >
                <SlidersHorizontal className="h-4 w-4" />
                Runtime
                {showDeploymentSettings ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </Button>
            )}

            {showNewChatAction && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={onClear}
                disabled={clearDisabled}
                className="rounded-full"
              >
                <Trash2 className="h-4 w-4" />
                New chat
              </Button>
            )}
          </div>
        </div>

        {chatMode === 'deployment' && showDeploymentSettings && (
          <div className="grid gap-3 rounded-2xl border border-border/80 bg-card/70 p-4 sm:grid-cols-2">
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Temperature</label>
              <Input
                type="number"
                step="0.1"
                min="0"
                value={deploymentTemperature}
                onChange={(event) => setDeploymentTemperature(event.target.value)}
                disabled={isStreaming || isUploadingImage}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Max New Tokens</label>
              <Input
                type="number"
                step="1"
                min="1"
                value={deploymentMaxNewTokens}
                onChange={(event) => setDeploymentMaxNewTokens(event.target.value)}
                disabled={isStreaming || isUploadingImage}
              />
            </div>
          </div>
        )}

        {showStarterPrompts && (
          <div className="flex flex-wrap gap-2">
            {starterPrompts.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => applyStarterPrompt(prompt)}
                className="rounded-full border border-border/80 bg-card/70 px-3 py-2 text-left text-xs text-muted-foreground transition-colors hover:border-primary/35 hover:bg-accent hover:text-foreground"
              >
                {prompt}
              </button>
            ))}
          </div>
        )}

        <div className="rounded-[28px] border border-border/80 bg-card/90 shadow-[0_14px_40px_rgba(0,0,0,0.24)]">
          {imageBlocks.length > 0 && (
            <div className="flex flex-wrap gap-2 px-4 pt-4">
              {imageBlocks.map((block, index) => (
                <div
                  key={`${block.image_path}-${index}`}
                  className="relative overflow-hidden rounded-lg border border-border/90 bg-background"
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

          <div className="px-4 pb-3 pt-4">
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
                    ? 'Ask MCP Tuna to investigate, plan, or act. You can attach images too.'
                    : 'Ask MCP Tuna to investigate, plan, or act.'
                  : selectedDeploymentId
                    ? selectedDeployment?.modality === 'vision-language'
                      ? 'Message the deployed VLM or attach images...'
                      : 'Message the deployed local model...'
                    : 'Select a running deployment first...'
              }
              className="min-h-[104px] w-full resize-none bg-transparent text-sm placeholder:text-muted-foreground focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>

          <div className="flex flex-wrap items-center gap-3 border-t border-border/60 px-4 py-3 sm:flex-nowrap">
            <div className="flex min-w-0 flex-1 items-center gap-2">
              {supportsImages && (
                <>
                  <button
                    type="button"
                    onClick={() => imageInputRef.current?.click()}
                    disabled={isStreaming || isUploadingImage || (chatMode === 'deployment' && !selectedDeploymentId)}
                    className="inline-flex h-9 items-center gap-2 rounded-full border border-input px-3 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                    title="Attach images"
                  >
                    <ImagePlus className="h-4 w-4" />
                    <span className="hidden sm:inline">Attach</span>
                  </button>
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
              <p className="hidden min-w-0 truncate text-[11px] text-muted-foreground xl:block">
                {selectionSummary}
              </p>
            </div>

            <div className="ml-auto flex shrink-0 items-center justify-end gap-2">
              <p className="hidden text-[11px] text-muted-foreground md:block">
                {supportsImages
                  ? 'Enter to send, Shift+Enter for a new line, attach to add images'
                  : 'Enter to send, Shift+Enter for a new line'}
              </p>
              {isStreaming ? (
                <Button
                  variant="destructive"
                  onClick={handleStop}
                  className="gap-2 rounded-full"
                  title="Stop generating"
                >
                  <Square className="h-4 w-4" />
                  Stop
                </Button>
              ) : (
                <Button
                  onClick={handleSubmit}
                  disabled={
                    (!input.trim() && imageBlocks.length === 0) ||
                    isUploadingImage ||
                    (chatMode === 'deployment' && !selectedDeploymentId)
                  }
                  className="gap-2 rounded-full"
                  title="Send message"
                >
                  <Send className="h-4 w-4" />
                  Send
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function resolveTemperature(value: string) {
  const parsed = Number.parseFloat(value.replace(',', '.'))
  return Number.isFinite(parsed) ? parsed : 0.7
}

function resolveMaxNewTokens(value: string) {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 512
}
