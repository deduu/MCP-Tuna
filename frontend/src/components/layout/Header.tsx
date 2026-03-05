import { useLocation } from 'react-router'
import { Search } from 'lucide-react'
import { useAppStore } from '@/stores/app'
import { Button } from '@/components/ui/button'

const ROUTE_TITLES: Record<string, string> = {
  '/': 'Dashboard',
  '/chat': 'Agent Chat',
  '/tools': 'Tool Explorer',
  '/pipeline': 'Pipeline Builder',
  '/datasets': 'Datasets',
  '/training': 'Training Jobs',
  '/deployments': 'Deployments',
  '/evaluation': 'Evaluation',
  '/settings': 'Settings',
}

export function Header() {
  const location = useLocation()
  const setCommandPaletteOpen = useAppStore((s) => s.setCommandPaletteOpen)

  const basePath = '/' + (location.pathname.split('/')[1] ?? '')
  const title = ROUTE_TITLES[basePath] ?? 'MCP Tuna'

  return (
    <header className="h-14 border-b bg-card flex items-center justify-between px-6">
      <h1 className="text-sm font-semibold">{title}</h1>

      <Button
        variant="outline"
        size="sm"
        className="gap-2 text-muted-foreground"
        onClick={() => setCommandPaletteOpen(true)}
      >
        <Search className="h-3.5 w-3.5" />
        <span className="text-xs">Search tools...</span>
        <kbd className="ml-2 pointer-events-none select-none rounded border bg-muted px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground">
          Ctrl+K
        </kbd>
      </Button>
    </header>
  )
}
