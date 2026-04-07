import { useState } from 'react'
import { MessageSquare, Rows3 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { CompareChatView } from './CompareChatView'
import { SingleChatView } from './SingleChatView'

const CHAT_VIEWS = [
  {
    id: 'single',
    label: 'Single Chat',
    icon: MessageSquare,
  },
  {
    id: 'compare',
    label: 'Compare',
    icon: Rows3,
  },
] as const

export function ChatPage() {
  const [activeTab, setActiveTab] = useState('single')

  return (
    <div className="relative flex h-[calc(100vh-3.5rem)] -m-6 flex-col overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-[radial-gradient(circle_at_top,rgba(59,130,246,0.12),transparent_68%)]" />

      <div className="relative flex justify-center border-b border-border/70 bg-background/80 px-6 py-3 backdrop-blur">
        <div className="inline-flex items-center gap-1 rounded-full border border-border/80 bg-card/85 p-1 shadow-sm shadow-black/20">
          {CHAT_VIEWS.map((view) => (
            <button
              key={view.id}
              type="button"
              onClick={() => setActiveTab(view.id)}
              className={cn(
                'inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm transition-colors',
                activeTab === view.id
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground',
              )}
            >
              <view.icon className="h-4 w-4" />
              <span>{view.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden px-6 py-4">
        {activeTab === 'single' ? <SingleChatView /> : <CompareChatView />}
      </div>
    </div>
  )
}
