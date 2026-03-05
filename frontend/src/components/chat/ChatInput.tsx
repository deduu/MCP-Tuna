import { useState, useRef, useCallback, type KeyboardEvent } from 'react'
import { Send, Square, ChevronDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useChatStore } from '@/stores/chat'
import { sendChatMessage } from '@/api/chat-client'

const AVAILABLE_MODELS = [
  { id: 'gpt-4o', label: 'GPT-4o' },
  { id: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4' },
  { id: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
]

export function ChatInput() {
  const [input, setInput] = useState('')
  const [model, setModel] = useState(AVAILABLE_MODELS[0].id)
  const [modelMenuOpen, setModelMenuOpen] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const abortController = useChatStore((s) => s.abortController)

  const selectedModel = AVAILABLE_MODELS.find((m) => m.id === model)

  const handleSubmit = useCallback(() => {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
    sendChatMessage(trimmed, { model })
  }, [input, isStreaming, model])

  const handleStop = useCallback(() => {
    abortController?.abort()
  }, [abortController])

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }

  return (
    <div className="border-t bg-card p-4">
      <div className="max-w-3xl mx-auto space-y-2">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder="Message MCP Tuna..."
              rows={1}
              disabled={isStreaming}
              className="w-full resize-none rounded-lg border border-input bg-background px-4 py-3 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-50"
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
              disabled={!input.trim()}
              className="shrink-0 self-end"
              title="Send message"
            >
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>

        <div className="flex items-center justify-between">
          {/* Model selector */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setModelMenuOpen(!modelMenuOpen)}
              className="flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded-md hover:bg-accent"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-primary" />
              {selectedModel?.label ?? model}
              <ChevronDown className="h-3 w-3" />
            </button>
            {modelMenuOpen && (
              <>
                <div
                  className="fixed inset-0 z-40"
                  onClick={() => setModelMenuOpen(false)}
                />
                <div className="absolute bottom-full mb-1 left-0 z-50 bg-popover border rounded-lg shadow-lg py-1 min-w-40">
                  {AVAILABLE_MODELS.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => {
                        setModel(m.id)
                        setModelMenuOpen(false)
                      }}
                      className={`w-full text-left px-3 py-1.5 text-xs hover:bg-accent transition-colors ${
                        m.id === model ? 'text-primary font-medium' : 'text-foreground'
                      }`}
                    >
                      {m.label}
                      <span className="block text-[10px] text-muted-foreground font-mono">
                        {m.id}
                      </span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          <p className="text-[10px] text-muted-foreground">
            Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  )
}
