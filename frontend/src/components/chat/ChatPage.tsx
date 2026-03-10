import { useEffect, useRef } from 'react'
import { User } from 'lucide-react'
import { useChatStore } from '@/stores/chat'
import { useToolCount } from '@/api/hooks/useToolRegistry'
import { AssistantMessage } from './AssistantMessage'
import { ChatInput } from './ChatInput'
import { Button } from '@/components/ui/button'
import { Trash2 } from 'lucide-react'

export function ChatPage() {
  const messages = useChatStore((s) => s.messages)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const chatMode = useChatStore((s) => s.chatMode)
  const selectedDeploymentId = useChatStore((s) => s.selectedDeploymentId)
  const clearMessages = useChatStore((s) => s.clearMessages)
  const { toolCount } = useToolCount()
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new content
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    // Only auto-scroll if user is near bottom (within 150px)
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 150
    if (isNearBottom) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages])

  return (
    <div className="flex flex-col h-[calc(100vh-3.5rem)] -m-6">
      {/* Header bar */}
      {messages.length > 0 && (
        <div className="flex items-center justify-end px-4 py-2 border-b">
          <Button
            variant="ghost"
            size="sm"
            onClick={clearMessages}
            disabled={isStreaming}
            className="text-muted-foreground gap-1.5"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </Button>
        </div>
      )}

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState
            chatMode={chatMode}
            selectedDeploymentId={selectedDeploymentId}
            toolCount={toolCount}
          />
        ) : (
          <div className="max-w-3xl mx-auto py-6 px-4 space-y-6">
            {messages.map((msg) =>
              msg.role === 'user' ? (
                <UserMessage key={msg.id} content={msg.content} />
              ) : (
                <AssistantMessage key={msg.id} message={msg} />
              ),
            )}
          </div>
        )}
      </div>

      {/* Input */}
      <ChatInput />
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

function EmptyState({
  chatMode,
  selectedDeploymentId,
  toolCount,
}: {
  chatMode: 'agent' | 'deployment'
  selectedDeploymentId: string | null
  toolCount: number
}) {
  const title = chatMode === 'agent' ? 'Agent Chat' : 'Deployed Local Chat'
  const description =
    chatMode === 'agent'
      ? `Chat with the MCP Tuna agent. It can use any of the ${toolCount || 'available'} tools to generate data, train models, deploy endpoints, and more.`
      : selectedDeploymentId
        ? 'Chat directly with the selected deployed model. This mode does not use MCP tools or agent planning.'
        : 'Select a running deployment below to chat directly with a local model without MCP tool use.'
  const detail =
    chatMode === 'agent'
      ? "You'll see the agent's thinking, tool calls, and decisions in real time."
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
