import { useEffect, useMemo, useRef, useState } from 'react'
import { Bot, History, MessageSquare, Pencil, Server, Trash2, User } from 'lucide-react'
import { useChatStore } from '@/stores/chat'
import { useToolCount } from '@/api/hooks/useToolRegistry'
import {
  useDeleteConversation,
  useDeploymentConversation,
  useDeploymentConversations,
  useDeployments,
  useRenameConversation,
} from '@/api/hooks/useDeployments'
import type { ChatContentBlock } from '@/lib/chat-content'
import {
  deleteAgentConversation,
  getAgentConversation,
  listAgentConversations,
  renameAgentConversation,
  type AgentConversationSummary,
  upsertAgentConversation,
} from '@/lib/agent-chat-history'
import { deploymentDisplayLabel } from '@/lib/compare-targets'
import { persistedConversationToChatMessages } from '@/lib/persisted-conversations'
import { cn, formatDateTime, formatTimeAgo } from '@/lib/utils'
import { AssistantMessage } from './AssistantMessage'
import { ChatInput } from './ChatInput'
import { MessageBlocks } from './MessageBlocks'
import { AVAILABLE_CHAT_MODELS } from './chat-model-options'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

export function SingleChatView() {
  const messages = useChatStore((s) => s.messages)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const chatMode = useChatStore((s) => s.chatMode)
  const selectedModel = useChatStore((s) => s.selectedModel)
  const selectedDeploymentId = useChatStore((s) => s.selectedDeploymentId)
  const deploymentConversationId = useChatStore((s) => s.deploymentConversationId)
  const clearMessages = useChatStore((s) => s.clearMessages)
  const replaceMessages = useChatStore((s) => s.replaceMessages)
  const setSelectedModel = useChatStore((s) => s.setSelectedModel)
  const { toolCount } = useToolCount()
  const { data: deployments = [] } = useDeployments()
  const renameConversation = useRenameConversation()
  const deleteConversation = useDeleteConversation()
  const [agentConversations, setAgentConversations] = useState<AgentConversationSummary[]>([])
  const [selectedAgentConversationId, setSelectedAgentConversationId] = useState<string | null>(null)
  const [selectedHistoryConversationId, setSelectedHistoryConversationId] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const selectedDeployment = deployments.find((deployment) => deployment.deployment_id === selectedDeploymentId) ?? null
  const { data: savedConversations = [] } = useDeploymentConversations(
    selectedDeploymentId ?? '',
    chatMode === 'deployment' && !!selectedDeploymentId,
  )
  const conversationToHydrate = useMemo(
    () =>
      selectedHistoryConversationId ??
      (chatMode === 'deployment' && messages.length === 0 ? deploymentConversationId : null),
    [chatMode, deploymentConversationId, messages.length, selectedHistoryConversationId],
  )
  const { data: hydratedConversation, isFetching: isHydratingConversation } = useDeploymentConversation(
    conversationToHydrate,
    chatMode === 'deployment' && !!conversationToHydrate,
  )
  const activeConversationSummary = savedConversations.find(
    (conversation) => conversation.conversation_id === (selectedHistoryConversationId ?? deploymentConversationId),
  )
  const activeAgentConversationSummary = agentConversations.find(
    (conversation) => conversation.conversation_id === selectedAgentConversationId,
  )

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 150
    if (isNearBottom) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages])

  useEffect(() => {
    if (chatMode !== 'deployment') {
      setSelectedHistoryConversationId(null)
      return
    }
    if (!selectedDeploymentId) {
      setSelectedHistoryConversationId(null)
    }
  }, [chatMode, selectedDeploymentId])

  useEffect(() => {
    if (!hydratedConversation) {
      return
    }

    replaceMessages(
      persistedConversationToChatMessages(hydratedConversation.messages),
      hydratedConversation.conversation_id,
    )
  }, [hydratedConversation, replaceMessages])

  useEffect(() => {
    setAgentConversations(listAgentConversations())
  }, [])

  useEffect(() => {
    if (chatMode !== 'agent') {
      setSelectedAgentConversationId(null)
      return
    }

    setAgentConversations(listAgentConversations())
  }, [chatMode])

  useEffect(() => {
    if (chatMode !== 'agent' || isStreaming || messages.length === 0) {
      return
    }

    const summary = upsertAgentConversation({
      conversationId: selectedAgentConversationId,
      model: selectedModel,
      messages,
    })
    if (summary.conversation_id !== selectedAgentConversationId) {
      setSelectedAgentConversationId(summary.conversation_id)
    }
    setAgentConversations(listAgentConversations())
  }, [chatMode, isStreaming, messages, selectedAgentConversationId, selectedModel])

  const handleClear = () => {
    setSelectedAgentConversationId(null)
    setSelectedHistoryConversationId(null)
    clearMessages()
  }

  const handleSelectAgentConversation = (conversationId: string) => {
    const conversation = getAgentConversation(conversationId)
    if (!conversation) {
      setAgentConversations(listAgentConversations())
      return
    }

    setSelectedAgentConversationId(conversation.conversation_id)
    if (conversation.model !== selectedModel) {
      setSelectedModel(conversation.model)
    }
    replaceMessages(conversation.messages, null)
  }

  const handleRenameAgentConversation = (conversationId: string, currentTitle?: string | null) => {
    const nextTitle = window.prompt('Rename conversation', currentTitle?.trim() || '')
    if (!nextTitle) return

    renameAgentConversation(conversationId, nextTitle)
    setAgentConversations(listAgentConversations())
  }

  const handleDeleteAgentConversation = (conversationId: string) => {
    if (!window.confirm('Delete this conversation and its saved messages?')) {
      return
    }

    deleteAgentConversation(conversationId)
    if (selectedAgentConversationId === conversationId) {
      handleClear()
      return
    }
    setAgentConversations(listAgentConversations())
  }

  const handleRenameConversation = (conversationId: string, currentTitle?: string | null) => {
    const nextTitle = window.prompt('Rename conversation', currentTitle?.trim() || '')
    if (!nextTitle) return

    renameConversation.mutate(
      { conversationId, title: nextTitle },
    )
  }

  const handleDeleteConversation = (conversationId: string) => {
    if (!window.confirm('Delete this conversation and its saved messages?')) {
      return
    }

    deleteConversation.mutate(conversationId, {
      onSuccess: () => {
        if ((selectedHistoryConversationId ?? deploymentConversationId) === conversationId) {
          handleClear()
        }
      },
    })
  }

  const activeConversationUpdatedAt =
    chatMode === 'agent'
      ? activeAgentConversationSummary?.updated_at
      : activeConversationSummary?.updated_at
  const activeConversationLabel = activeConversationUpdatedAt
    ? formatTimeAgo(activeConversationUpdatedAt) ?? formatDateTime(activeConversationUpdatedAt)
    : null
  const selectedModelLabel =
    AVAILABLE_CHAT_MODELS.find((model) => model.id === selectedModel)?.label ?? selectedModel
  const sessionContext =
    chatMode === 'agent'
      ? `${selectedModelLabel} with ${toolCount} tools available for agent work.`
      : selectedDeployment
        ? selectedDeployment.modality === 'vision-language'
          ? `${deploymentDisplayLabel(selectedDeployment)} in multimodal deployment mode.`
          : `${deploymentDisplayLabel(selectedDeployment)} in direct deployment mode.`
        : 'Select a running deployment below to start a direct model session.'
  const clearDisabled = isStreaming || (messages.length === 0 && !deploymentConversationId)
  const selectedDeploymentConversationId = selectedHistoryConversationId ?? deploymentConversationId

  return (
    <div className="grid h-full min-h-0 gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
      <aside className="h-full space-y-3 overflow-y-auto rounded-2xl border border-border/80 bg-card/70 p-4 shadow-sm shadow-black/20">
        {chatMode === 'agent' ? (
          <>
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <History className="h-4 w-4 text-muted-foreground" />
                <h2 className="text-sm font-semibold">Sessions</h2>
              </div>
              <Badge variant="outline">{agentConversations.length}</Badge>
            </div>

            <p className="text-xs text-muted-foreground">
              Resume earlier tool-agent chats saved in this browser.
            </p>

            <button
              type="button"
              onClick={handleClear}
              disabled={isStreaming}
              className={cn(
                'w-full rounded-lg border px-3 py-2 text-left transition-colors',
                !selectedAgentConversationId && messages.length === 0
                  ? 'border-primary/45 bg-primary/15'
                  : 'border-border/90 bg-secondary hover:border-primary/40 hover:bg-accent',
              )}
            >
              <div className="flex items-center gap-2">
                <MessageSquare className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-sm font-medium">New chat</span>
              </div>
              <p className="mt-1 text-[11px] text-muted-foreground">
                Start a fresh agent session with the current model.
              </p>
            </button>

            {agentConversations.length === 0 ? (
              <div className="rounded-lg border border-dashed px-3 py-4 text-xs text-muted-foreground">
                Your agent sessions will appear here after the first completed reply.
              </div>
            ) : (
              <div className="space-y-2">
                {agentConversations.map((conversation) => {
                  const isSelected = conversation.conversation_id === selectedAgentConversationId
                  const updatedLabel =
                    formatTimeAgo(conversation.updated_at) ??
                    formatDateTime(conversation.updated_at) ??
                    'unknown'

                  return (
                    <button
                      key={conversation.conversation_id}
                      type="button"
                      onClick={() => handleSelectAgentConversation(conversation.conversation_id)}
                      disabled={isStreaming}
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
                              handleRenameAgentConversation(conversation.conversation_id, conversation.title)
                            }}
                            className="rounded p-1 text-muted-foreground hover:bg-background/70 hover:text-foreground"
                            title="Rename conversation"
                          >
                            <Pencil className="h-3 w-3" />
                          </button>
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation()
                              handleDeleteAgentConversation(conversation.conversation_id)
                            }}
                            className="rounded p-1 text-muted-foreground hover:bg-background/70 hover:text-destructive"
                            title="Delete conversation"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                      <div className="mt-1 flex items-center gap-2 text-[11px] text-muted-foreground">
                        <span className="truncate">{getChatModelLabel(conversation.model)}</span>
                      </div>
                      <p className="mt-1 text-[11px] text-muted-foreground">
                        Updated {updatedLabel}
                      </p>
                    </button>
                  )
                })}
              </div>
            )}
          </>
        ) : (
          <>
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <History className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-semibold">Saved Conversations</h2>
            </div>
            <Badge variant="outline">{savedConversations.length}</Badge>
          </div>

          {selectedDeployment ? (
            <p className="text-xs text-muted-foreground">
              Resume persisted chats for {selectedDeployment.name?.trim() || selectedDeployment.model_path.split('/').pop() || selectedDeployment.model_path}.
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">
              Select a running deployment to browse persisted chat history.
            </p>
          )}

            <button
              type="button"
              onClick={handleClear}
              disabled={isStreaming}
              className={cn(
                'w-full rounded-lg border px-3 py-2 text-left transition-colors',
                !selectedHistoryConversationId && !deploymentConversationId
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

          {!selectedDeploymentId ? (
            <div className="rounded-lg border border-dashed px-3 py-4 text-xs text-muted-foreground">
              No deployment selected.
            </div>
          ) : savedConversations.length === 0 ? (
            <div className="rounded-lg border border-dashed px-3 py-4 text-xs text-muted-foreground">
              No persisted conversations yet for this deployment.
            </div>
          ) : (
            <div className="space-y-2">
              {savedConversations.map((conversation) => {
                const isSelected = conversation.conversation_id === selectedDeploymentConversationId
                const updatedLabel =
                  formatTimeAgo(conversation.updated_at) ??
                  formatDateTime(conversation.updated_at) ??
                  'unknown'

                return (
                  <button
                    key={conversation.conversation_id}
                    type="button"
                    onClick={() => setSelectedHistoryConversationId(conversation.conversation_id)}
                    disabled={isStreaming || renameConversation.isPending || deleteConversation.isPending}
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
                          className="rounded p-1 text-muted-foreground hover:bg-background/70 hover:text-foreground"
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
                          className="rounded p-1 text-muted-foreground hover:bg-background/70 hover:text-destructive"
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
              })}
            </div>
          )}
          </>
        )}
      </aside>

      <div className="flex h-full min-h-0 flex-col">
        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
          {isHydratingConversation ? (
            <div className="flex h-full items-center justify-center px-4 text-sm text-muted-foreground">
              Loading saved conversation...
            </div>
          ) : messages.length === 0 ? (
            <EmptyState
              chatMode={chatMode}
              selectedDeploymentLabel={
                selectedDeployment ? deploymentDisplayLabel(selectedDeployment) : null
              }
              selectedDeploymentModality={selectedDeployment?.modality ?? 'text'}
              selectedModelLabel={selectedModelLabel}
            />
          ) : (
            <>
              <div className="mx-auto flex w-full max-w-3xl items-center justify-between gap-3 px-4 pt-4">
                <p className="min-w-0 truncate text-xs text-muted-foreground">{sessionContext}</p>
                <div className="flex items-center gap-3">
                  {chatMode === 'deployment' && activeConversationLabel && (
                    <span className="hidden text-[11px] text-muted-foreground sm:inline">
                      Updated {activeConversationLabel}
                    </span>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleClear}
                    disabled={clearDisabled}
                    className="gap-1.5 text-muted-foreground"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    {chatMode === 'deployment' ? 'New chat' : 'Clear chat'}
                  </Button>
                </div>
              </div>
              <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
                {messages.map((msg) =>
                  msg.role === 'user' ? (
                    msg.parts ? (
                      <UserMessageWithParts key={msg.id} parts={msg.parts} />
                    ) : (
                      <UserMessage key={msg.id} content={msg.content} />
                    )
                  ) : (
                    <AssistantMessage key={msg.id} message={msg} />
                  ),
                )}
              </div>
            </>
          )}
        </div>

        <ChatInput onClear={handleClear} clearDisabled={clearDisabled} />
      </div>
    </div>
  )
}

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex gap-3 max-w-3xl">
      <div className="shrink-0 mt-1">
        <div className="h-7 w-7 rounded-full bg-secondary flex items-center justify-center">
          <User className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>
      <div className="text-sm leading-relaxed whitespace-pre-wrap pt-1">
        {content}
      </div>
    </div>
  )
}

function UserMessageWithParts({ parts }: { parts: ChatContentBlock[] }) {
  return (
    <div className="flex gap-3 max-w-3xl">
      <div className="shrink-0 mt-1">
        <div className="h-7 w-7 rounded-full bg-secondary flex items-center justify-center">
          <User className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>
      <MessageBlocks blocks={parts} className="pt-1" />
    </div>
  )
}

function EmptyState({
  chatMode,
  selectedDeploymentLabel,
  selectedDeploymentModality,
  selectedModelLabel,
}: {
  chatMode: 'agent' | 'deployment'
  selectedDeploymentLabel: string | null
  selectedDeploymentModality: 'text' | 'vision-language' | 'unknown'
  selectedModelLabel: string
}) {
  const title = chatMode === 'agent' ? 'How can MCP Tuna help?' : 'Talk to a deployment directly'
  const description =
    chatMode === 'agent'
      ? `Start with ${selectedModelLabel} in tool-agent mode. The workspace stays quiet until you ask for investigation, planning, or action.`
      : selectedDeploymentLabel
        ? selectedDeploymentModality === 'vision-language'
          ? `Use ${selectedDeploymentLabel} for direct multimodal testing with text and images, without the agent orchestration layer.`
          : `Use ${selectedDeploymentLabel} for direct model testing without MCP tool execution.`
        : 'Select a running deployment below when you want a quieter, direct model conversation.'

  return (
    <div className="flex h-full items-center justify-center px-4 py-12">
      <div className="w-full max-w-2xl space-y-5 text-center">
        <div className="flex justify-center">
          <Badge variant={chatMode === 'agent' ? 'success' : 'warning'} className="gap-1.5">
            {chatMode === 'agent' ? <Bot className="h-3 w-3" /> : <Server className="h-3 w-3" />}
            {chatMode === 'agent' ? selectedModelLabel : selectedDeploymentLabel ?? 'Deployment mode'}
          </Badge>
        </div>
        <div className="space-y-3">
          <h2 className="text-3xl font-semibold tracking-tight">{title}</h2>
          <p className="mx-auto max-w-xl text-sm leading-relaxed text-muted-foreground">{description}</p>
        </div>
      </div>
    </div>
  )
}

function getChatModelLabel(modelId: string) {
  return AVAILABLE_CHAT_MODELS.find((model) => model.id === modelId)?.label ?? modelId
}
