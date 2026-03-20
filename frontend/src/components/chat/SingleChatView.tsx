import { useEffect, useMemo, useRef, useState } from 'react'
import { History, MessageSquare, Trash2, User } from 'lucide-react'
import { useChatStore } from '@/stores/chat'
import { useToolCount } from '@/api/hooks/useToolRegistry'
import {
  useDeploymentConversation,
  useDeploymentConversations,
  useDeployments,
} from '@/api/hooks/useDeployments'
import type { ChatContentBlock } from '@/lib/chat-content'
import { persistedConversationToChatMessages } from '@/lib/persisted-conversations'
import { cn, formatDateTime, formatTimeAgo } from '@/lib/utils'
import { AssistantMessage } from './AssistantMessage'
import { ChatInput } from './ChatInput'
import { MessageBlocks } from './MessageBlocks'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

export function SingleChatView() {
  const messages = useChatStore((s) => s.messages)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const chatMode = useChatStore((s) => s.chatMode)
  const selectedDeploymentId = useChatStore((s) => s.selectedDeploymentId)
  const deploymentConversationId = useChatStore((s) => s.deploymentConversationId)
  const clearMessages = useChatStore((s) => s.clearMessages)
  const replaceMessages = useChatStore((s) => s.replaceMessages)
  const { toolCount } = useToolCount()
  const { data: deployments = [] } = useDeployments()
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

  const handleClear = () => {
    setSelectedHistoryConversationId(null)
    clearMessages()
  }

  const activeConversationLabel = activeConversationSummary?.updated_at
    ? formatTimeAgo(activeConversationSummary.updated_at) ?? formatDateTime(activeConversationSummary.updated_at)
    : null

  return (
    <div className={cn('h-full min-h-0', chatMode === 'deployment' ? 'grid lg:grid-cols-[280px_minmax(0,1fr)]' : 'flex flex-col')}>
      {chatMode === 'deployment' && (
        <aside className="border-b lg:border-b-0 lg:border-r bg-secondary/10 p-4 space-y-3 overflow-y-auto">
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
                ? 'border-primary bg-primary/10'
                : 'border-border hover:border-primary/40 hover:bg-background/60',
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
                const isSelected =
                  conversation.conversation_id === (selectedHistoryConversationId ?? deploymentConversationId)
                const updatedLabel =
                  formatTimeAgo(conversation.updated_at) ??
                  formatDateTime(conversation.updated_at) ??
                  'unknown'

                return (
                  <button
                    key={conversation.conversation_id}
                    type="button"
                    onClick={() => setSelectedHistoryConversationId(conversation.conversation_id)}
                    disabled={isStreaming}
                    className={cn(
                      'w-full rounded-lg border px-3 py-2 text-left transition-colors',
                      isSelected
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:border-primary/40 hover:bg-background/60',
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-sm font-medium">
                        {conversation.title?.trim() || conversation.conversation_id}
                      </span>
                      <Badge variant="outline">{conversation.message_count}</Badge>
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
        </aside>
      )}

      <div className="flex h-full min-h-0 flex-col">
        {(messages.length > 0 || (chatMode === 'deployment' && deploymentConversationId)) && (
          <div className="flex items-center justify-between gap-3 px-4 py-2 border-b">
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
              {chatMode === 'deployment' && deploymentConversationId && (
                <Badge variant="outline" className="font-mono text-[10px]">
                  {deploymentConversationId}
                </Badge>
              )}
              {chatMode === 'deployment' && activeConversationLabel && (
                <span>Persisted history updated {activeConversationLabel}</span>
              )}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClear}
              disabled={isStreaming}
              className="text-muted-foreground gap-1.5"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear
            </Button>
          </div>
        )}

        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {isHydratingConversation ? (
            <div className="flex h-full items-center justify-center px-4 text-sm text-muted-foreground">
              Loading saved conversation…
            </div>
          ) : messages.length === 0 ? (
            <EmptyState
              chatMode={chatMode}
              selectedDeploymentId={selectedDeploymentId}
              selectedDeploymentModality={selectedDeployment?.modality ?? 'text'}
              toolCount={toolCount}
            />
          ) : (
            <div className="max-w-3xl mx-auto py-6 px-4 space-y-6">
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
          )}
        </div>

        <ChatInput />
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
  selectedDeploymentId,
  selectedDeploymentModality,
  toolCount,
}: {
  chatMode: 'agent' | 'deployment'
  selectedDeploymentId: string | null
  selectedDeploymentModality: 'text' | 'vision-language' | 'unknown'
  toolCount: number
}) {
  const title = chatMode === 'agent' ? 'Agent Chat' : 'Deployed Local Chat'
  const description =
    chatMode === 'agent'
      ? `Chat with the MCP Tuna agent. It can use any of the ${toolCount || 'available'} tools to generate data, train models, deploy endpoints, and more.`
      : selectedDeploymentId
        ? selectedDeploymentModality === 'vision-language'
          ? 'Chat directly with the selected deployed vision-language model. You can attach images and text together in this mode.'
          : 'Chat directly with the selected deployed model. This mode does not use MCP tools or agent planning.'
        : 'Select a running deployment below to chat directly with a local model without MCP tool use.'
  const detail =
    chatMode === 'agent'
      ? "You'll see the agent's thinking, tool calls, and decisions in real time. Images are forwarded as structured multimodal message blocks."
      : 'Use this mode when you want a manually deployed local model to stay fully under your own resource control.'

  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="space-y-3 max-w-md">
        <h2 className="text-lg font-semibold">{title}</h2>
        <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
        <p className="text-xs text-muted-foreground">{detail}</p>
      </div>
    </div>
  )
}
