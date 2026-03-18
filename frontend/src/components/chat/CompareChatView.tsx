import { useEffect, useMemo, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import { Bot, ChevronDown, ChevronUp, ImagePlus, PencilLine, Plus, Server, Square, Trash2, X } from 'lucide-react'
import { toast } from 'sonner'
import { useDeployments } from '@/api/hooks/useDeployments'
import type { Deployment } from '@/api/types'
import {
  runDeploymentCompareTarget,
  serializeCompareMessages,
  streamAgentCompareTarget,
} from '@/api/chat-compare-client'
import {
  buildUserChatContent,
  extractTextFromChatContent,
  type ChatImageBlock,
} from '@/lib/chat-content'
import { resolveCompareTarget, shortDeploymentLabel } from '@/lib/compare-targets'
import { uploadAsset } from '@/lib/uploads'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ComparePane } from './ComparePane'
import { CompareTargetConfigurator } from './CompareTargetConfigurator'
import { AVAILABLE_CHAT_MODELS } from './chat-model-options'
import { useChatCompareStore, type CompareMetrics, type CompareTargetConfig } from '@/stores/chatCompare'

function createAgentTarget(index: number): CompareTargetConfig {
  const model = AVAILABLE_CHAT_MODELS[index % AVAILABLE_CHAT_MODELS.length] ?? AVAILABLE_CHAT_MODELS[0]
  return {
    id: crypto.randomUUID(),
    kind: 'agent',
    label: model.label,
    model: model.id,
  }
}

function createDeploymentTarget(deployment: Deployment): CompareTargetConfig {
  return {
    id: crypto.randomUUID(),
    kind: 'deployment',
    label: shortDeploymentLabel(deployment.model_path),
    deploymentId: deployment.deployment_id,
    deploymentLabel: shortDeploymentLabel(deployment.model_path),
    deploymentModality: deployment.modality ?? 'text',
  }
}

export function CompareChatView() {
  const sessions = useChatCompareStore((state) => state.sessions)
  const baselineTargetId = useChatCompareStore((state) => state.baselineTargetId)
  const addTarget = useChatCompareStore((state) => state.addTarget)
  const updateTarget = useChatCompareStore((state) => state.updateTarget)
  const removeTarget = useChatCompareStore((state) => state.removeTarget)
  const setBaselineTargetId = useChatCompareStore((state) => state.setBaselineTargetId)
  const addUserMessage = useChatCompareStore((state) => state.addUserMessage)
  const startAssistantMessage = useChatCompareStore((state) => state.startAssistantMessage)
  const appendToken = useChatCompareStore((state) => state.appendToken)
  const addThinking = useChatCompareStore((state) => state.addThinking)
  const addToolStart = useChatCompareStore((state) => state.addToolStart)
  const addToolEnd = useChatCompareStore((state) => state.addToolEnd)
  const addReflection = useChatCompareStore((state) => state.addReflection)
  const setMessageMetrics = useChatCompareStore((state) => state.setMessageMetrics)
  const finishAssistantMessage = useChatCompareStore((state) => state.finishAssistantMessage)
  const failAssistantMessage = useChatCompareStore((state) => state.failAssistantMessage)
  const setConversationId = useChatCompareStore((state) => state.setConversationId)
  const setAbortController = useChatCompareStore((state) => state.setAbortController)
  const clearTargetMessages = useChatCompareStore((state) => state.clearTargetMessages)
  const clearAllMessages = useChatCompareStore((state) => state.clearAllMessages)

  const [input, setInput] = useState('')
  const [imageBlocks, setImageBlocks] = useState<ChatImageBlock[]>([])
  const [isUploadingImage, setIsUploadingImage] = useState(false)
  const [isSetupCollapsed, setIsSetupCollapsed] = useState(false)
  const { data: deployments = [] } = useDeployments()

  const runningDeployments = useMemo(
    () => deployments.filter((deployment) => deployment.status === 'running'),
    [deployments],
  )

  useEffect(() => {
    if (sessions.length > 0) {
      return
    }
    addTarget(createAgentTarget(0))
    addTarget(createAgentTarget(1))
  }, [addTarget, sessions.length])

  const resolvedSessions = useMemo(
    () =>
      sessions.map((session) => ({
        ...session,
        target: resolveCompareTarget(session.target, runningDeployments),
      })),
    [runningDeployments, sessions],
  )

  const isRunning = resolvedSessions.some((session) => session.status === 'streaming')
  const anyMessages = resolvedSessions.some((session) => session.messages.length > 0)
  const allTargetsSupportImages =
    resolvedSessions.length > 0 &&
    resolvedSessions.every(
      (session) =>
        session.target.kind === 'agent' || session.target.deploymentModality === 'vision-language',
    )
  const allTargetsConfigured =
    resolvedSessions.length > 0 &&
    resolvedSessions.every((session) =>
      session.target.kind === 'agent' ? Boolean(session.target.model) : Boolean(session.target.deploymentId),
    )

  useEffect(() => {
    if (allTargetsConfigured) {
      setIsSetupCollapsed(true)
    }
  }, [allTargetsConfigured])

  const gridClassName = useMemo(() => {
    if (sessions.length <= 1) return 'grid-cols-1'
    if (sessions.length === 2) return 'grid-cols-1 xl:grid-cols-2'
    if (sessions.length === 3) return 'grid-cols-1 md:grid-cols-2 2xl:grid-cols-3'
    return 'grid-cols-1 md:grid-cols-2 2xl:grid-cols-4'
  }, [resolvedSessions.length])

  const baselineSession =
    resolvedSessions.find((session) => session.target.id === baselineTargetId) ?? null
  const baselineMetrics = getLatestAssistantMetrics(baselineSession?.messages ?? [])

  async function handleSubmit() {
    const trimmed = input.trim()
    if ((!trimmed && imageBlocks.length === 0) || isRunning || isUploadingImage) {
      return
    }
    if (resolvedSessions.length === 0) {
      toast.error('Add at least one compare target first')
      return
    }
    if (imageBlocks.length > 0 && !allTargetsSupportImages) {
      toast.error('Image attachments require every compare target to support multimodal chat')
      return
    }

    const payload = buildUserChatContent(trimmed, imageBlocks)
    const userText = extractTextFromChatContent(payload)

    for (const session of resolvedSessions) {
      if (session.target.kind === 'agent' && !session.target.model) {
        toast.error(`Select a model for ${session.target.label}`)
        return
      }
      if (session.target.kind === 'deployment' && !session.target.deploymentId) {
        toast.error(`Select a deployment for ${session.target.label}`)
        return
      }
    }

    for (const session of resolvedSessions) {
      addUserMessage(session.target.id, {
        content: userText,
        parts: Array.isArray(payload) ? payload : undefined,
      })
      const assistantMessageId = startAssistantMessage(session.target.id)
      const controller = new AbortController()
      setAbortController(session.target.id, controller)

      if (session.target.kind === 'agent') {
        const currentSession = useChatCompareStore
          .getState()
          .sessions.find((candidate) => candidate.target.id === session.target.id)

        if (!currentSession) {
          continue
        }

        void streamAgentCompareTarget(
          session.target,
          serializeCompareMessages(currentSession.messages),
          {
            signal: controller.signal,
          },
          {
            onToken: (token) => appendToken(session.target.id, assistantMessageId, token),
            onThinking: (content) => addThinking(session.target.id, assistantMessageId, content),
            onToolStart: (tool, args) => addToolStart(session.target.id, assistantMessageId, tool, args),
            onToolEnd: (tool, durationMs) => addToolEnd(session.target.id, assistantMessageId, tool, durationMs),
            onReflection: (isReady, explanation) =>
              addReflection(session.target.id, assistantMessageId, isReady, explanation),
            onMetrics: (metrics) => setMessageMetrics(session.target.id, assistantMessageId, metrics),
            onComplete: ({ metrics, modelId }) =>
              finishAssistantMessage(session.target.id, assistantMessageId, {
                metrics,
                modelId: modelId ?? session.target.model ?? null,
              }),
            onError: (message) => failAssistantMessage(session.target.id, assistantMessageId, message),
          },
        ).catch((error) => {
          if (error instanceof DOMException && error.name === 'AbortError') {
            finishAssistantMessage(session.target.id, assistantMessageId)
            return
          }
          failAssistantMessage(
            session.target.id,
            assistantMessageId,
            error instanceof Error ? error.message : 'Compare request failed',
          )
        })
        continue
      }

      void runDeploymentCompareTarget(
        session.target,
        payload,
        session.conversationId,
        controller.signal,
        {
          onToken: (token) => appendToken(session.target.id, assistantMessageId, token),
          onComplete: ({ response, metrics, conversationId, modelId }) => {
            if (conversationId) {
              setConversationId(session.target.id, conversationId)
            }
            finishAssistantMessage(session.target.id, assistantMessageId, {
              content: response,
              metrics,
              modelId: modelId ?? session.target.deploymentLabel ?? null,
            })
          },
          onError: (message) => failAssistantMessage(session.target.id, assistantMessageId, message),
        },
      )
    }

    setInput('')
    setImageBlocks([])
  }

  function handleStop() {
    for (const session of resolvedSessions) {
      session.abortController?.abort()
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
    setImageBlocks((current) => current.filter((_, currentIndex) => currentIndex !== index))
  }

  function handleAddDeploymentTarget() {
    const deployment = runningDeployments[0]
    if (!deployment) {
      toast.error('No running deployments are available for compare mode')
      return
    }
    addTarget(createDeploymentTarget(deployment))
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void handleSubmit()
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
        <div className="rounded-2xl border bg-card/95">
          <div className="px-5 py-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="space-y-1">
                <h2 className="text-lg font-semibold">Compare Mode</h2>
                <p className="text-sm text-muted-foreground">
                  Compare multiple agents or deployments side by side, then judge quality against a
                  baseline with latency, token, and cost deltas inline.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button variant="outline" size="sm" onClick={() => addTarget(createAgentTarget(sessions.length))}>
                  <Plus className="h-4 w-4" />
                  Add agent
                </Button>
                <Button variant="outline" size="sm" onClick={handleAddDeploymentTarget}>
                  <Plus className="h-4 w-4" />
                  Add deployment
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsSetupCollapsed((current) => !current)}
                  disabled={resolvedSessions.length === 0}
                >
                  {isSetupCollapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
                  {isSetupCollapsed ? 'Edit targets' : 'Minimize setup'}
                </Button>
                <Button variant="ghost" size="sm" onClick={clearAllMessages} disabled={isRunning || !anyMessages}>
                  <Trash2 className="h-4 w-4" />
                  Clear all
                </Button>
                <Button variant="destructive" size="sm" onClick={handleStop} disabled={!isRunning}>
                  <Square className="h-4 w-4" />
                  Stop
                </Button>
              </div>
            </div>
          </div>

          {isSetupCollapsed ? (
            <div className="border-t px-5 py-4">
              <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
                {resolvedSessions.map((session, index) => (
                  <button
                    key={session.target.id}
                    type="button"
                    onClick={() => setIsSetupCollapsed(false)}
                    className="rounded-xl border border-border/70 bg-secondary/15 px-4 py-3 text-left transition-colors hover:bg-secondary/25"
                  >
                    <div className="flex items-center gap-2 text-sm font-medium">
                      {session.target.kind === 'agent' ? (
                        <Bot className="h-4 w-4 text-primary" />
                      ) : (
                        <Server className="h-4 w-4 text-amber-400" />
                      )}
                      <span className="truncate">{session.target.label}</span>
                      {session.target.id === baselineTargetId && <Badge variant="success">Baseline</Badge>}
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <span>Target {index + 1}</span>
                      <span className="rounded-full bg-background px-2 py-0.5">
                        {session.target.kind === 'agent'
                          ? session.target.model
                          : session.target.deploymentLabel ?? session.target.deploymentId ?? 'Not selected'}
                      </span>
                      {session.target.kind === 'deployment' &&
                        session.target.deploymentModality === 'vision-language' && (
                          <Badge variant="warning">VLM</Badge>
                        )}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="border-t p-4">
              <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
                {sessions.map((session, index) => (
                  <CompareTargetConfigurator
                    key={session.target.id}
                    session={session}
                    index={index}
                    baselineTargetId={baselineTargetId}
                    runningDeployments={runningDeployments}
                    onSetBaseline={setBaselineTargetId}
                    onUpdate={updateTarget}
                    onRemove={removeTarget}
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        <div className={cn('grid gap-4', gridClassName)}>
          {resolvedSessions.map((session) => (
            <ComparePane
              key={session.target.id}
              session={session}
              baselineSession={baselineSession}
              baselineMetrics={baselineMetrics}
              isBaseline={session.target.id === baselineTargetId}
              onClear={() => clearTargetMessages(session.target.id)}
              disabled={isRunning}
            />
          ))}
        </div>
      </div>

      <div className="mx-auto w-full max-w-5xl rounded-2xl border bg-card/95 p-4 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline">{resolvedSessions.length} targets</Badge>
            <Badge variant={allTargetsSupportImages ? 'success' : 'secondary'}>
              {allTargetsSupportImages ? 'Images supported' : 'Text compare only'}
            </Badge>
            {isSetupCollapsed && (
              <Button variant="ghost" size="sm" onClick={() => setIsSetupCollapsed(false)}>
                <PencilLine className="h-4 w-4" />
                Edit targets
              </Button>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            Use baseline selection above to measure latency, token, and tool deltas per pane.
          </p>
        </div>

        <div className="mt-3 space-y-3">
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

          <div className="flex gap-3">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={isRunning || isUploadingImage || resolvedSessions.length === 0}
              placeholder="Send one prompt to every selected compare target..."
              className="min-h-[74px] flex-1 resize-none rounded-xl border border-input bg-background px-4 py-3 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            />
            <div className="flex shrink-0 items-end gap-2">
              {allTargetsSupportImages && (
                <>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() => document.getElementById('compare-image-input')?.click()}
                    disabled={isRunning || isUploadingImage || resolvedSessions.length === 0}
                    title="Attach images"
                  >
                    <ImagePlus className="h-4 w-4" />
                  </Button>
                  <input
                    id="compare-image-input"
                    type="file"
                    className="hidden"
                    accept="image/*"
                    multiple
                    onChange={handlePickImage}
                    disabled={isRunning || isUploadingImage}
                  />
                </>
              )}
              <Button
                onClick={() => void handleSubmit()}
                disabled={
                  (!input.trim() && imageBlocks.length === 0) ||
                  resolvedSessions.length === 0 ||
                  isRunning ||
                  isUploadingImage ||
                  (imageBlocks.length > 0 && !allTargetsSupportImages)
                }
                className="min-w-[112px]"
              >
                Compare
              </Button>
            </div>
          </div>

          <div className="flex items-center justify-between gap-3">
            <p className="text-[11px] text-muted-foreground">
              {allTargetsSupportImages
                ? 'Enter to compare, Shift+Enter for a new line, image button to attach'
                : 'Enter to compare, Shift+Enter for a new line'}
            </p>
            {!allTargetsSupportImages && imageBlocks.length === 0 && (
              <p className="text-[11px] text-muted-foreground">
                Add only agent targets or VLM deployments if you want image comparison.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function getLatestAssistantMetrics(
  messages: Array<{ role: 'user' | 'assistant'; metrics?: CompareMetrics | null; isStreaming?: boolean }>,
) {
  const lastAssistant = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant' && !message.isStreaming && message.metrics)
  return lastAssistant?.metrics ?? null
}
