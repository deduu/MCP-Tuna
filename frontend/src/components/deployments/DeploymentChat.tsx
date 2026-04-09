import { useEffect, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  ChevronDown,
  ChevronUp,
  History,
  ImagePlus,
  MessageSquare,
  Pencil,
  Send,
  SlidersHorizontal,
  Trash2,
  X,
} from 'lucide-react'
import {
  useDeleteConversation,
  useDeploymentConversation,
  useDeploymentConversations,
  useRenameConversation,
} from '@/api/hooks/useDeployments'
import type { ConversationMessage, Deployment, ThinkingMode } from '@/api/types'
import { mcpCall } from '@/api/client'
import { streamDeploymentTextChat } from '@/api/deployment-chat-stream'
import type { ChatContentBlock, ChatImageBlock } from '@/lib/chat-content'
import {
  buildUserChatContent,
  extractTextFromChatContent,
  sanitizeChatContentForRequest,
} from '@/lib/chat-content'
import {
  DEFAULT_DEPLOYMENT_CHAT_SETTINGS,
  getDeploymentChatDraftSettings,
  setDeploymentChatDraftSettings,
  type DeploymentChatDraftSettings,
} from '@/lib/deployment-chat-settings'
import { THINKING_MODE_OPTIONS } from '@/lib/thinking-mode'
import { uploadAsset } from '@/lib/uploads'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { MessageBlocks } from '@/components/chat/MessageBlocks'
import { toast } from 'sonner'
import { cn, formatDateTime, formatTimeAgo } from '@/lib/utils'

type DeploymentChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  parts?: ChatContentBlock[]
  error?: boolean
  isStreaming?: boolean
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

function resolveDeploymentChatDraftSettings(deployment: Deployment): DeploymentChatDraftSettings {
  const persisted = getDeploymentChatDraftSettings(deployment.deployment_id)
  return {
    ...DEFAULT_DEPLOYMENT_CHAT_SETTINGS,
    ...persisted,
    systemPrompt: persisted?.systemPrompt ?? deployment.system_prompt ?? '',
    thinkingMode:
      deployment.modality === 'text'
        ? persisted?.thinkingMode ?? deployment.thinking_mode ?? 'default'
        : 'default',
  }
}

export function DeploymentChat({ deployment }: DeploymentChatProps) {
  const initialSettings = resolveDeploymentChatDraftSettings(deployment)
  const [messages, setMessages] = useState<DeploymentChatMessage[]>([])
  const [input, setInput] = useState('')
  const [systemPrompt, setSystemPrompt] = useState(initialSettings.systemPrompt)
  const [imageBlocks, setImageBlocks] = useState<ChatImageBlock[]>([])
  const [isUploadingImage, setIsUploadingImage] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [selectedHistoryConversationId, setSelectedHistoryConversationId] = useState<string | null>(null)
  const [isTextStreaming, setIsTextStreaming] = useState(false)
  const [thinkingMode, setThinkingMode] = useState<ThinkingMode>(initialSettings.thinkingMode)
  const [temperature, setTemperature] = useState(initialSettings.temperature)
  const [topP, setTopP] = useState(initialSettings.topP)
  const [topK, setTopK] = useState(initialSettings.topK)
  const [maxNewTokens, setMaxNewTokens] = useState(initialSettings.maxNewTokens)
  const [showHistory, setShowHistory] = useState(true)
  const [showSettings, setShowSettings] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const textAbortControllerRef = useRef<AbortController | null>(null)
  const renameConversation = useRenameConversation()
  const deleteConversation = useDeleteConversation()
  const { data: savedConversations = [] } = useDeploymentConversations(deployment.deployment_id, true)
  const { data: selectedConversation, isFetching: isLoadingConversation } = useDeploymentConversation(
    selectedHistoryConversationId,
    Boolean(selectedHistoryConversationId),
  )

  const vlmChatMutation = useMutation<HostChatResult, Error, string>({
    mutationFn: async (message) =>
      mcpCall<HostChatResult>('host.chat_vlm', {
        deployment_id: deployment.deployment_id,
        messages: [
          {
            role: 'user',
            content: sanitizeChatContentForRequest(buildUserChatContent(message, imageBlocks)),
          },
        ],
        temperature: resolveTemperature(temperature),
        top_p: resolveTopP(topP),
        top_k: resolveTopK(topK),
        max_new_tokens: resolveMaxNewTokens(maxNewTokens),
        ...(systemPrompt.trim() ? { system_prompt: systemPrompt.trim() } : {}),
        ...(conversationId ? { conversation_id: conversationId } : {}),
      }),
    onSuccess: (result) => {
      setConversationId(result.conversation_id)
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'assistant', content: result.response },
      ])
      clearImageBlocks()
    },
    onError: (error) => {
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'assistant', content: error.message, error: true },
      ])
    },
  })

  const isPending = vlmChatMutation.isPending || isTextStreaming

  useEffect(() => {
    const el = scrollRef.current
    if (!el) {
      return
    }
    el.scrollTop = el.scrollHeight
  }, [messages, isPending])

  useEffect(() => {
    const restored = resolveDeploymentChatDraftSettings(deployment)
    textAbortControllerRef.current?.abort()
    setMessages([])
    setInput('')
    setSystemPrompt(restored.systemPrompt)
    setThinkingMode(restored.thinkingMode)
    setTemperature(restored.temperature)
    setTopP(restored.topP)
    setTopK(restored.topK)
    setMaxNewTokens(restored.maxNewTokens)
    setConversationId(null)
    setSelectedHistoryConversationId(null)
    setIsTextStreaming(false)
    setShowHistory(true)
    setShowSettings(false)
    clearImageBlocks()
    vlmChatMutation.reset()
  }, [deployment.deployment_id])

  useEffect(() => {
    if (!selectedConversation) {
      return
    }

    setConversationId(selectedConversation.conversation_id)
    setSystemPrompt(selectedConversation.system_prompt ?? deployment.system_prompt ?? '')
    setThinkingMode(selectedConversation.thinking_mode ?? deployment.thinking_mode ?? 'default')
    setMessages(selectedConversation.messages.map(toDeploymentChatMessage))
  }, [deployment.system_prompt, deployment.thinking_mode, selectedConversation])

  const handleSubmit = () => {
    const trimmed = input.trim()
    if ((!trimmed && imageBlocks.length === 0) || isPending || isUploadingImage || deployment.status !== 'running') {
      return
    }

    const userContent = buildUserChatContent(trimmed, imageBlocks)
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: trimmed,
        parts: deployment.modality === 'vision-language' && Array.isArray(userContent) ? userContent : undefined,
      },
    ])
    setInput('')

    if (deployment.modality === 'vision-language') {
      vlmChatMutation.mutate(trimmed)
      return
    }

    void streamTextDeploymentMessage(trimmed)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSubmit()
    }
  }

  const handleClear = () => {
    if (isPending) {
      return
    }
    setMessages([])
    setInput('')
    setConversationId(null)
    setSelectedHistoryConversationId(null)
    clearImageBlocks()
    vlmChatMutation.reset()
  }

  const handleRenameConversation = (conversationId: string, currentTitle?: string | null) => {
    const nextTitle = window.prompt('Rename conversation', currentTitle?.trim() || '')
    if (!nextTitle) return
    renameConversation.mutate({ conversationId, title: nextTitle })
  }

  const handleDeleteConversation = (targetConversationId: string) => {
    if (!window.confirm('Delete this conversation and its saved messages?')) {
      return
    }
    const activeConversationId = selectedHistoryConversationId ?? conversationId
    deleteConversation.mutate(targetConversationId, {
      onSuccess: () => {
        if (activeConversationId === targetConversationId) {
          handleClear()
        }
      },
    })
  }

  const subtitle =
    deployment.type === 'api'
      ? deployment.modality === 'vision-language'
        ? 'Messages are sent through the deployed VLM API runtime.'
        : 'Messages are streamed through the deployed API runtime.'
      : deployment.modality === 'vision-language'
        ? 'Messages use the live deployed VLM runtime managed by the gateway.'
        : 'Messages stream from the live deployed model runtime managed by the gateway.'

  const activeConversationUpdatedAt =
    selectedConversation?.updated_at ??
    savedConversations.find((conversation) => conversation.conversation_id === conversationId)?.updated_at

  const activeConversationLabel = activeConversationUpdatedAt
    ? `${formatTimeAgo(activeConversationUpdatedAt) ?? 'recently'}`
    : null

  function persistDraftSettings(patch: Partial<DeploymentChatDraftSettings>) {
    setDeploymentChatDraftSettings(deployment.deployment_id, {
      systemPrompt,
      thinkingMode: deployment.modality === 'text' ? thinkingMode : 'default',
      temperature,
      topP,
      topK,
      maxNewTokens,
      ...patch,
    })
  }

  async function streamTextDeploymentMessage(message: string) {
    const assistantId = crypto.randomUUID()
    const abortController = new AbortController()
    textAbortControllerRef.current = abortController
    setIsTextStreaming(true)
    setMessages((current) => [
      ...current,
      { id: assistantId, role: 'assistant', content: '', isStreaming: true },
    ])

    try {
      await streamDeploymentTextChat(
        {
          deployment_id: deployment.deployment_id,
          message,
          conversation_id: conversationId,
          temperature: resolveTemperature(temperature),
          top_p: resolveTopP(topP),
          top_k: resolveTopK(topK),
          max_new_tokens: resolveMaxNewTokens(maxNewTokens),
          thinking_mode: thinkingMode,
          prefer_runtime_metrics: deployment.type === 'api',
          ...(systemPrompt.trim() ? { system_prompt: systemPrompt.trim() } : {}),
          signal: abortController.signal,
        },
        {
          onToken: (token) => {
            setMessages((current) =>
              current.map((currentMessage) =>
                currentMessage.id === assistantId
                  ? {
                      ...currentMessage,
                      content: currentMessage.content + token,
                    }
                  : currentMessage,
              ),
            )
          },
          onComplete: (result) => {
            setConversationId(result.conversation_id)
            setMessages((current) =>
              current.map((currentMessage) =>
                currentMessage.id === assistantId
                  ? {
                      ...currentMessage,
                      content: result.response || currentMessage.content,
                      isStreaming: false,
                    }
                  : currentMessage,
              ),
            )
            clearImageBlocks()
          },
          onError: (messageText) => {
            throw new Error(messageText)
          },
        },
      )
    } catch (error) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        const messageText = error instanceof Error ? error.message : 'Deployment chat failed'
        setMessages((current) =>
          current.map((currentMessage) =>
            currentMessage.id === assistantId
              ? {
                  ...currentMessage,
                  content: messageText,
                  error: true,
                  isStreaming: false,
                }
              : currentMessage,
          ),
        )
      }
    } finally {
      textAbortControllerRef.current = null
      setIsTextStreaming(false)
    }
  }

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

  function clearImageBlocks() {
    setImageBlocks((current) => {
      for (const block of current) {
        if (block.preview_url) {
          URL.revokeObjectURL(block.preview_url)
        }
      }
      return []
    })
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-4 rounded-2xl border border-border/80 bg-muted/30 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                className={cn(
                  deployment.type === 'api'
                    ? 'border-transparent bg-primary/20 text-primary'
                    : 'border-transparent bg-[var(--color-ns-host)]/20 text-[var(--color-ns-host)]',
                )}
              >
                {deployment.type === 'api' ? 'API Runtime' : 'MCP Runtime'}
              </Badge>
              <Badge variant={deployment.status === 'running' ? 'success' : 'secondary'}>
                {deployment.status}
              </Badge>
              <Badge variant="outline">
                {deployment.modality === 'vision-language' ? 'Images enabled' : 'Text only'}
              </Badge>
            </div>

            <div className="space-y-1">
              <h3 className="text-lg font-semibold tracking-tight">Deployment Chat</h3>
              <p className="text-sm text-muted-foreground">{subtitle}</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {conversationId && (
              <Badge variant="outline" className="font-mono text-[10px]">
                {conversationId}
              </Badge>
            )}
            {activeConversationLabel && (
              <span className="text-[11px] text-muted-foreground">Updated {activeConversationLabel}</span>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowHistory((current) => !current)}
              disabled={isPending}
              className="gap-1.5 rounded-full"
            >
              <History className="h-4 w-4" />
              History
              {showHistory ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSettings((current) => !current)}
              disabled={isPending}
              className="gap-1.5 rounded-full"
            >
              <SlidersHorizontal className="h-4 w-4" />
              Settings
              {showSettings ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </Button>
            <Button variant="ghost" size="sm" onClick={handleClear} disabled={isPending} className="gap-1.5">
              <Trash2 className="h-4 w-4" />
              Clear
            </Button>
          </div>
        </div>
      </div>

      <div className={cn('grid gap-4', showHistory ? 'lg:grid-cols-[280px_minmax(0,1fr)]' : 'grid-cols-1')}>
        {showHistory && (
          <div className="rounded-xl border border-border/90 bg-muted p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
            <div className="mb-3 flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <History className="h-4 w-4 text-muted-foreground" />
                <p className="text-sm font-medium">Saved Conversations</p>
              </div>
              <Badge variant="outline">{savedConversations.length}</Badge>
            </div>

            <div className="space-y-2">
              <button
                type="button"
                onClick={handleClear}
                disabled={isPending}
                className={cn(
                  'w-full rounded-lg border px-3 py-2 text-left transition-colors',
                  !selectedHistoryConversationId && !conversationId
                    ? 'border-primary/45 bg-primary/15'
                    : 'border-border/90 bg-secondary hover:border-primary/40 hover:bg-accent',
                )}
              >
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="text-sm font-medium">New conversation</span>
                </div>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Clear the current transcript and start fresh.
                </p>
              </button>

              {savedConversations.length === 0 ? (
                <p className="rounded-lg border border-dashed border-border/80 bg-secondary px-3 py-4 text-xs text-muted-foreground">
                  No persisted conversations yet for this deployment.
                </p>
              ) : (
                savedConversations.map((conversation) => {
                  const isSelected = conversation.conversation_id === (selectedHistoryConversationId ?? conversationId)
                  const updatedLabel =
                    formatTimeAgo(conversation.updated_at) ??
                    formatDateTime(conversation.updated_at) ??
                    'unknown'

                  return (
                    <button
                      key={conversation.conversation_id}
                      type="button"
                      onClick={() => setSelectedHistoryConversationId(conversation.conversation_id)}
                      disabled={isPending || renameConversation.isPending || deleteConversation.isPending}
                      className={cn(
                        'w-full rounded-lg border px-3 py-2 text-left transition-colors',
                        isSelected
                          ? 'border-primary/45 bg-primary/15'
                          : 'border-border/90 bg-secondary hover:border-primary/40 hover:bg-accent',
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-sm font-medium">
                          {conversation.title?.trim() || conversation.conversation_id}
                        </span>
                        <div className="flex items-center gap-1">
                          <Badge variant="outline">{conversation.message_count}</Badge>
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation()
                              handleRenameConversation(conversation.conversation_id, conversation.title)
                            }}
                            className="rounded p-1 text-muted-foreground hover:bg-card hover:text-foreground"
                            title="Rename conversation"
                          >
                            <Pencil className="h-3 w-3" />
                          </button>
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation()
                              handleDeleteConversation(conversation.conversation_id)
                            }}
                            className="rounded p-1 text-muted-foreground hover:bg-card hover:text-destructive"
                            title="Delete conversation"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                      {conversation.title && (
                        <p className="mt-1 font-mono text-[10px] text-muted-foreground">
                          {conversation.conversation_id}
                        </p>
                      )}
                      <p className="mt-1 text-[11px] text-muted-foreground">
                        Updated {updatedLabel}
                      </p>
                    </button>
                  )
                })
              )}
            </div>
          </div>
        )}

        <div className="space-y-4">
            <div
              ref={scrollRef}
              className="min-h-[360px] max-h-[56vh] overflow-y-auto rounded-[28px] border border-border/80 bg-muted/30 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]"
            >
              {isLoadingConversation ? (
                <p className="text-sm text-muted-foreground">Loading saved conversation...</p>
              ) : messages.length === 0 ? (
                <div className="flex h-full items-center justify-center text-center">
                  <div className="max-w-md space-y-2">
                    <p className="text-2xl font-semibold tracking-tight">Talk to this deployment directly</p>
                    <p className="text-sm text-muted-foreground">
                      Use this panel to validate direct responses from the deployed model without
                      the agent layer in between.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="mx-auto max-w-3xl space-y-4">
                  {messages.map((message) => (
                    <div key={message.id} className="space-y-1.5">
                      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                        {message.role === 'user' ? 'User' : message.error ? 'Error' : 'Assistant'}
                      </div>
                      {message.parts ? (
                        <MessageBlocks blocks={message.parts} />
                      ) : (
                        <div
                          className={cn(
                            'whitespace-pre-wrap rounded-xl border px-4 py-3 text-sm leading-6',
                            message.error
                              ? 'border-destructive/50 bg-destructive/10 text-destructive'
                              : message.role === 'user'
                                ? 'border-border/90 bg-secondary'
                                : 'border-border/90 bg-card',
                            message.isStreaming && !message.content && 'text-muted-foreground',
                          )}
                        >
                          {message.content || (message.isStreaming ? 'Generating response...' : '')}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {showSettings && (
              <div className="rounded-xl border border-border/90 bg-secondary p-4 shadow-sm shadow-black/20">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">Temperature</label>
                  <Input
                    type="number"
                    step="0.1"
                    min="0"
                    value={temperature}
                    onChange={(event) => {
                      setTemperature(event.target.value)
                      persistDraftSettings({ temperature: event.target.value })
                    }}
                    disabled={isPending}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">Top P</label>
                  <Input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    value={topP}
                    onChange={(event) => {
                      setTopP(event.target.value)
                      persistDraftSettings({ topP: event.target.value })
                    }}
                    disabled={isPending}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">Top K</label>
                  <Input
                    type="number"
                    step="1"
                    min="1"
                    value={topK}
                    onChange={(event) => {
                      setTopK(event.target.value)
                      persistDraftSettings({ topK: event.target.value })
                    }}
                    disabled={isPending}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">Max New Tokens</label>
                  <Input
                    type="number"
                    step="1"
                    min="1"
                    value={maxNewTokens}
                    onChange={(event) => {
                      setMaxNewTokens(event.target.value)
                      persistDraftSettings({ maxNewTokens: event.target.value })
                    }}
                    disabled={isPending}
                  />
                </div>
              </div>

              {deployment.modality === 'text' && (
                <div className="mt-4 space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">Thinking Mode</label>
                  <select
                    value={thinkingMode}
                    onChange={(event) => {
                      const nextThinkingMode = event.target.value as ThinkingMode
                      setThinkingMode(nextThinkingMode)
                      persistDraftSettings({ thinkingMode: nextThinkingMode })
                    }}
                    disabled={isPending}
                    className="flex h-9 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {THINKING_MODE_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <p className="text-[11px] text-muted-foreground">
                    Default follows the deployment default and, for saved threads, the conversation setting already persisted by the backend. Clear the conversation to guarantee a clean reset.
                  </p>
                </div>
              )}

              <div className="mt-4 space-y-1">
                <label className="text-xs font-medium text-muted-foreground">System Prompt</label>
                <textarea
                  value={systemPrompt}
                  onChange={(event) => {
                    setSystemPrompt(event.target.value)
                    persistDraftSettings({ systemPrompt: event.target.value })
                  }}
                  rows={4}
                  disabled={isPending}
                  placeholder="Optional. Use this to mirror notebook-style system instructions."
                  className="flex min-h-[96px] w-full resize-y rounded-lg border border-input bg-background px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                />
                <p className="text-[11px] text-muted-foreground">
                  Applied to new or cleared conversations. If you change it mid-thread, clear the conversation to guarantee a fresh system prompt.
                </p>
              </div>
              </div>
            )}

            <div className="rounded-[28px] border border-border/80 bg-card/90 p-3 shadow-[0_14px_40px_rgba(0,0,0,0.22)]">
              {deployment.modality === 'vision-language' && imageBlocks.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-2 px-1">
                  {imageBlocks.map((block, index) => (
                    <div key={`${block.image_path}-${index}`} className="relative overflow-hidden rounded-lg border border-border/90 bg-secondary">
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

              <div className="flex gap-2">
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={3}
                  disabled={isPending || isUploadingImage || deployment.status !== 'running'}
                  placeholder={
                    deployment.status === 'running'
                      ? deployment.modality === 'vision-language'
                        ? 'Ask the deployed vision-language model a question or attach images...'
                        : 'Ask the deployed model a question...'
                      : 'Start or redeploy the model to continue this conversation.'
                  }
                  className="flex min-h-[104px] w-full resize-none rounded-xl border border-input bg-background px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                />

                <div className="flex shrink-0 items-end gap-2">
                  {deployment.modality === 'vision-language' && (
                    <>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => imageInputRef.current?.click()}
                        disabled={isPending || isUploadingImage || deployment.status !== 'running'}
                        className="gap-2"
                      >
                        <ImagePlus className="h-4 w-4" />
                        <span className="hidden sm:inline">Attach</span>
                      </Button>
                      <input
                        ref={imageInputRef}
                        type="file"
                        className="hidden"
                        accept="image/*"
                        multiple
                        onChange={handlePickImage}
                        disabled={isPending || isUploadingImage}
                      />
                    </>
                  )}
                  <Button
                    onClick={handleSubmit}
                    disabled={(!input.trim() && imageBlocks.length === 0) || isPending || isUploadingImage || deployment.status !== 'running'}
                    className="gap-2 rounded-full"
                  >
                    <Send className="h-4 w-4" />
                    {isPending ? 'Sending...' : 'Send'}
                  </Button>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-border/60 pt-3">
                <p className="text-[11px] text-muted-foreground">
                  {deployment.modality === 'vision-language'
                    ? 'Enter to send, Shift+Enter for a new line, attach images when needed.'
                    : 'Enter to send, Shift+Enter for a new line.'}
                </p>
                <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                  <span>
                    {showSettings
                      ? `Max ${resolveMaxNewTokens(maxNewTokens)} tokens`
                      : 'Open Settings for temperature and system prompt'}
                  </span>
                  {deployment.status !== 'running' && <Badge variant="warning">View-only while stopped</Badge>}
                </div>
              </div>
            </div>
          </div>
        </div>
    </div>
  )
}

function toDeploymentChatMessage(message: ConversationMessage): DeploymentChatMessage {
  const parts = parseStructuredContent(message.content)
  return {
    id: `saved-${message.sequence}`,
    role: message.role,
    content: parts ? extractTextFromChatContent(parts) : typeof message.content === 'string' ? message.content : '',
    parts,
  }
}

function parseStructuredContent(content: ConversationMessage['content']): ChatContentBlock[] | undefined {
  if (!Array.isArray(content)) {
    return undefined
  }

  const blocks: ChatContentBlock[] = []
  for (const item of content) {
    if (typeof item !== 'object' || item === null) {
      continue
    }
    if (item.type === 'text' && typeof item.text === 'string') {
      blocks.push({ type: 'text', text: item.text })
      continue
    }
    if (item.type === 'image_path' && typeof item.image_path === 'string') {
      blocks.push({ type: 'image_path', image_path: item.image_path })
    }
  }

  return blocks.length > 0 ? blocks : undefined
}

function resolveTemperature(value: string) {
  const parsed = Number.parseFloat(value.replace(',', '.'))
  return Number.isFinite(parsed) ? parsed : 0.7
}

function resolveTopP(value: string) {
  const parsed = Number.parseFloat(value.replace(',', '.'))
  if (!Number.isFinite(parsed)) return 0.95
  return Math.max(0, Math.min(1, parsed))
}

function resolveTopK(value: string) {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 50
}

function resolveMaxNewTokens(value: string) {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 512
}
