import { useLocation, useNavigate } from 'react-router'
import { ArrowLeft, Search } from 'lucide-react'
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
  const navigate = useNavigate()
  const setCommandPaletteOpen = useAppStore((s) => s.setCommandPaletteOpen)

  const basePath = '/' + (location.pathname.split('/')[1] ?? '')
  const title = ROUTE_TITLES[basePath] ?? 'MCP Tuna'
  const canGoBack = basePath !== '/'

  const handleBack = () => {
    const historyIndex = typeof window !== 'undefined'
      ? Number(window.history.state?.idx ?? 0)
      : 0

    if (historyIndex > 0) {
      navigate(-1)
      return
    }

    navigate('/')
  }

  return (
    <header className="h-14 border-b bg-card flex items-center justify-between px-6">
      <div className="flex items-center gap-3">
        {canGoBack && (
          <Button
            variant="ghost"
            size="sm"
            className="gap-2 text-muted-foreground"
            onClick={handleBack}
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            <span className="text-xs">Back</span>
          </Button>
        )}
        <h1 className="text-sm font-semibold">{title}</h1>
      </div>

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
