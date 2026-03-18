import { NavLink } from 'react-router'
import { cn } from '@/lib/utils'
import { useAppStore } from '@/stores/app'
import {
  LayoutDashboard,
  MessageSquare,
  Wrench,
  GitBranch,
  Database,
  Cpu,
  Rocket,
  BarChart3,
  Settings,
  PanelLeftClose,
  Fish,
} from 'lucide-react'
import { Button } from '@/components/ui/button'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/chat', label: 'Chat', icon: MessageSquare },
  { to: '/tools', label: 'Tools', icon: Wrench },
  { to: '/pipeline', label: 'Pipeline', icon: GitBranch },
  { to: '/datasets', label: 'Datasets', icon: Database },
  { to: '/training', label: 'Training', icon: Cpu },
  { to: '/deployments', label: 'Deployments', icon: Rocket },
  { to: '/evaluation', label: 'Evaluation', icon: BarChart3 },
]

export function Sidebar() {
  const collapsed = useAppStore((s) => s.sidebarCollapsed)
  const toggleSidebar = useAppStore((s) => s.toggleSidebar)

  return (
    <aside
      className={cn(
        'flex flex-col h-screen border-r bg-card transition-all duration-200',
        collapsed ? 'w-[72px]' : 'w-56',
      )}
    >
      <div
        className={cn(
          'flex items-center border-b',
          collapsed ? 'justify-center px-2 py-3' : 'justify-between gap-3 p-4',
        )}
      >
        {collapsed ? (
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSidebar}
            className="h-9 w-9 text-primary hover:text-primary"
            aria-label="Expand sidebar"
            title="Expand sidebar"
          >
            <Fish className="h-5 w-5 shrink-0" />
          </Button>
        ) : (
          <>
            <div className="flex items-center gap-2 min-w-0">
              <Fish className="h-6 w-6 text-primary shrink-0" />
              <span className="font-bold text-sm tracking-tight truncate">
                MCP Tuna
              </span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleSidebar}
              className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
              aria-label="Collapse sidebar"
              title="Collapse sidebar"
            >
              <PanelLeftClose className="h-4 w-4" />
            </Button>
          </>
        )}
      </div>

      <nav className="flex-1 py-2 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-4 py-2 text-sm transition-colors hover:bg-accent',
                isActive
                  ? 'text-primary bg-primary/10'
                  : 'text-muted-foreground',
                collapsed && 'justify-center px-0',
              )
            }
          >
            <Icon className="h-4 w-4 shrink-0" />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="border-t p-2">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            cn(
              'flex items-center gap-3 px-4 py-2 text-sm transition-colors hover:bg-accent rounded-md',
              isActive ? 'text-primary' : 'text-muted-foreground',
              collapsed && 'justify-center px-0',
            )
          }
        >
          <Settings className="h-4 w-4 shrink-0" />
          {!collapsed && <span>Settings</span>}
        </NavLink>
      </div>
    </aside>
  )
}
