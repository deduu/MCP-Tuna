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
  PanelLeft,
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
        collapsed ? 'w-16' : 'w-56',
      )}
    >
      <div className="flex items-center gap-2 p-4 border-b">
        <Fish className="h-6 w-6 text-primary shrink-0" />
        {!collapsed && (
          <span className="font-bold text-sm tracking-tight">MCP Tuna</span>
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
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleSidebar}
          className={cn('w-full mt-1', collapsed ? '' : 'justify-start px-4')}
        >
          {collapsed ? (
            <PanelLeft className="h-4 w-4" />
          ) : (
            <>
              <PanelLeftClose className="h-4 w-4" />
              <span className="text-xs text-muted-foreground ml-2">Collapse</span>
            </>
          )}
        </Button>
      </div>
    </aside>
  )
}
