import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Send, Trash2 } from 'lucide-react'
import type { Deployment } from '@/api/types'
import { mcpCall } from '@/api/client'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

type DeploymentChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
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
  const [conversationId, setConversationId] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const chatMutation = useMutation<HostChatResult, Error, string>({
    onMutate: (message) => {
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'user', content: message },
      ])
    },
    mutationFn: async (message) =>
      mcpCall<HostChatResult>('host.chat', {
        deployment_id: deployment.deployment_id,
        message,
        ...(conversationId ? { conversation_id: conversationId } : {}),
      }),
    onSuccess: (result) => {
      setConversationId(result.conversation_id)
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'assistant', content: result.response },
      ])
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
    if (!trimmed || chatMutation.isPending || deployment.status !== 'running') {
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
    setMessages([])
    setInput('')
    setConversationId(null)
    chatMutation.reset()
  }

  const subtitle =
    deployment.type === 'api'
      ? 'Messages are sent through the deployed API runtime.'
      : 'Messages use the live deployed model runtime managed by the gateway.'

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
                  <div className={message.error ? 'text-sm text-destructive whitespace-pre-wrap' : 'text-sm whitespace-pre-wrap'}>
                    {message.content}
                  </div>
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
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
            disabled={chatMutation.isPending || deployment.status !== 'running'}
            placeholder={
              deployment.status === 'running'
                ? 'Ask the deployed model a question...'
                : 'Start or redeploy the model to chat with it.'
            }
            className="flex min-h-[96px] w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
          />
          <div className="flex items-center justify-between gap-3">
            <p className="text-[11px] text-muted-foreground">
              Enter to send, Shift+Enter for a new line
            </p>
            <Button onClick={handleSubmit} disabled={!input.trim() || chatMutation.isPending || deployment.status !== 'running'}>
              <Send className="h-4 w-4" />
              {chatMutation.isPending ? 'Sending...' : 'Send'}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
