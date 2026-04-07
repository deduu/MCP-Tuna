import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

interface LogViewerProps {
  logs: string[]
  isLoading?: boolean
}

export function LogViewer({ logs, isLoading }: LogViewerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  if (isLoading) {
    return (
      <div className="max-h-80 overflow-y-auto rounded-xl border border-border/90 bg-[#0b1016] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
        <p className="text-sm font-mono text-muted-foreground animate-pulse">Loading logs...</p>
      </div>
    )
  }

  if (logs.length === 0) {
    return (
      <div className="rounded-xl border border-border/90 bg-[#0b1016] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
        <p className="text-sm font-mono text-muted-foreground">No logs available</p>
      </div>
    )
  }

  return (
    <div className="max-h-80 overflow-y-auto rounded-xl border border-border/90 bg-[#0b1016] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
      {logs.map((line, i) => (
        <p key={i} className={cn('text-xs font-mono leading-5', colorForLogLine(line))}>
          {line}
        </p>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

function colorForLogLine(line: string): string {
  const lower = line.toLowerCase()
  if (lower.includes('error') || lower.includes('fatal')) return 'text-red-400'
  if (lower.includes('warn')) return 'text-amber-400'
  if (lower.includes('info')) return 'text-blue-400'
  if (/^\d{4}-\d{2}-\d{2}|^\[[\d:.\-T]+\]/.test(line)) return 'text-muted-foreground'
  return 'text-foreground/80'
}
