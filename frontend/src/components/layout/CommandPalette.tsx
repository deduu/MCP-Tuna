import { useEffect, useMemo } from 'react'
import { Command } from 'cmdk'
import { useNavigate } from 'react-router'
import { useAppStore } from '@/stores/app'
import { useToolCount, useToolRegistry } from '@/api/hooks/useToolRegistry'
import { NAMESPACE_MAP, getNamespaceFromToolName, getToolShortName } from '@/lib/tool-registry'

export function CommandPalette() {
  const open = useAppStore((s) => s.commandPaletteOpen)
  const setOpen = useAppStore((s) => s.setCommandPaletteOpen)
  const navigate = useNavigate()
  const { data: tools } = useToolRegistry()
  const { toolCount } = useToolCount()

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen(!open)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, setOpen])

  const groupedTools = useMemo(() => {
    if (!tools) return {}
    const groups: Record<string, typeof tools> = {}
    for (const tool of tools) {
      const ns = getNamespaceFromToolName(tool.name)
      ;(groups[ns] ??= []).push(tool)
    }
    return groups
  }, [tools])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
      <div className="fixed inset-0 bg-black/60" onClick={() => setOpen(false)} />
      <Command
        className="relative w-full max-w-lg rounded-xl border bg-popover text-popover-foreground shadow-2xl"
        loop
      >
        <Command.Input
          placeholder={toolCount > 0 ? `Search ${toolCount} tools...` : 'Search tools...'}
          className="w-full border-b bg-transparent px-4 py-3 text-sm outline-none placeholder:text-muted-foreground"
          autoFocus
        />
        <Command.List className="max-h-72 overflow-y-auto p-2">
          <Command.Empty className="py-6 text-center text-sm text-muted-foreground">
            No tools found.
          </Command.Empty>

          {Object.entries(groupedTools).map(([ns, nsTools]) => {
            const nsInfo = NAMESPACE_MAP[ns]
            return (
              <Command.Group key={ns} heading={nsInfo?.label ?? ns}>
                {nsTools.map((tool) => (
                  <Command.Item
                    key={tool.name}
                    value={`${tool.name} ${tool.description}`}
                    onSelect={() => {
                      setOpen(false)
                      navigate(`/tools/${ns}/${getToolShortName(tool.name)}`)
                    }}
                    className="flex items-center gap-3 px-3 py-2 text-sm rounded-md cursor-pointer aria-selected:bg-accent"
                  >
                    <span
                      className="h-2 w-2 rounded-full shrink-0"
                      style={{ backgroundColor: nsInfo?.color }}
                    />
                    <div className="flex-1 min-w-0">
                      <span className="font-medium">{getToolShortName(tool.name)}</span>
                      <span className="ml-2 text-xs text-muted-foreground truncate">
                        {tool.description?.slice(0, 60)}
                      </span>
                    </div>
                  </Command.Item>
                ))}
              </Command.Group>
            )
          })}

          <Command.Group heading="Navigation">
            {[
              { label: 'Dashboard', path: '/' },
              { label: 'Chat', path: '/chat' },
              { label: 'Pipeline Builder', path: '/pipeline' },
              { label: 'Datasets', path: '/datasets' },
              { label: 'Training', path: '/training' },
              { label: 'Deployments', path: '/deployments' },
            ].map((item) => (
              <Command.Item
                key={item.path}
                value={item.label}
                onSelect={() => {
                  setOpen(false)
                  navigate(item.path)
                }}
                className="px-3 py-2 text-sm rounded-md cursor-pointer aria-selected:bg-accent"
              >
                {item.label}
              </Command.Item>
            ))}
          </Command.Group>
        </Command.List>
      </Command>
    </div>
  )
}
