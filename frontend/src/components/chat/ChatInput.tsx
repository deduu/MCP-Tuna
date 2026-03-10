import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import { Send, Square } from 'lucide-react'
import { sendChatMessage } from '@/api/chat-client'
import { useDeployments } from '@/api/hooks/useDeployments'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useChatStore } from '@/stores/chat'

const AVAILABLE_MODELS = [
  { id: 'gpt-4o', label: 'GPT-4o' },
  { id: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4' },
  { id: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
]

export function ChatInput() {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

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

  const helperText =
    chatMode === 'agent'
      ? 'Tool Agent uses managed providers and can call MCP tools.'
      : selectedDeployment
        ? `Deployed Local chats directly with ${shortDeploymentLabel(selectedDeployment.model_path)}. MCP tools are disabled in this mode.`
        : 'Deployed Local needs a running deployment. Start one from Deployments first.'

  const handleSubmit = useCallback(() => {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) {
      return
    }

    if (chatMode === 'deployment' && !selectedDeploymentId) {
      return
    }

    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    void sendChatMessage(trimmed, {
      source: chatMode,
      model: selectedModel,
      deploymentId: selectedDeploymentId,
    })
  }, [chatMode, input, isStreaming, selectedDeploymentId, selectedModel])

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
                    </option>
                  ))
                )}
              </select>
              <Badge variant="warning">Direct model chat</Badge>
            </>
          )}
        </div>

        <div className="flex gap-2">
          <div className="relative flex-1">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={isStreaming || (chatMode === 'deployment' && !selectedDeploymentId)}
              placeholder={
                chatMode === 'agent'
                  ? 'Message MCP Tuna...'
                  : selectedDeploymentId
                    ? 'Message the deployed local model...'
                    : 'Select a running deployment first...'
              }
              className="w-full resize-none rounded-lg border border-input bg-background px-4 py-3 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>
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
              disabled={!input.trim() || (chatMode === 'deployment' && !selectedDeploymentId)}
              className="shrink-0 self-end"
              title="Send message"
            >
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>

        <div className="flex items-center justify-between gap-3">
          <p className="text-[11px] text-muted-foreground">{helperText}</p>
          <p className="text-[10px] text-muted-foreground">Enter to send, Shift+Enter for new line</p>
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
