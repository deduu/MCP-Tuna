import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useSetupCheck } from '@/api/hooks/useSystemResources'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Stethoscope, ChevronDown, ChevronUp, Play, HardDrive, Sparkles } from 'lucide-react'

interface CheckItem {
  name: string
  status: 'pass' | 'warn' | 'fail'
  detail?: string
}

function CollapsibleResult({
  label,
  isOpen,
  onToggle,
  children,
}: {
  label: string
  isOpen: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="flex items-center justify-between w-full px-3 py-2 text-sm text-muted-foreground hover:bg-secondary/50 transition-colors cursor-pointer"
      >
        <span>{label}</span>
        {isOpen ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
      </button>
      {isOpen && <div className="px-3 pb-3 border-t">{children}</div>}
    </div>
  )
}

function CheckList({ checks }: { checks: CheckItem[] }) {
  return (
    <div className="space-y-1.5 pt-2">
      {checks.map((check) => (
        <div key={check.name} className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">{check.name}</span>
          <div className="flex items-center gap-2">
            {check.detail && (
              <span className="text-xs text-muted-foreground font-mono">{check.detail}</span>
            )}
            <Badge
              variant={
                check.status === 'pass' ? 'success' : check.status === 'warn' ? 'warning' : 'error'
              }
            >
              {check.status}
            </Badge>
          </div>
        </div>
      ))}
    </div>
  )
}

export function DiagnosticsSection() {
  const { data: setup, isLoading: setupLoading } = useSetupCheck()

  const preflight = useToolExecution()
  const diskPreflight = useToolExecution()
  const prescribe = useToolExecution()

  const [openSections, setOpenSections] = useState<Record<string, boolean>>({})

  const toggle = (key: string) =>
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }))

  const runPreflight = () => {
    preflight.mutate(
      { toolName: 'system.preflight_check', args: {} },
      { onSuccess: () => setOpenSections((prev) => ({ ...prev, preflight: true })) },
    )
  }

  const runDisk = () => {
    diskPreflight.mutate(
      { toolName: 'system.disk_preflight', args: {} },
      { onSuccess: () => setOpenSections((prev) => ({ ...prev, disk: true })) },
    )
  }

  const runPrescribe = () => {
    prescribe.mutate(
      { toolName: 'system.prescribe', args: {} },
      { onSuccess: () => setOpenSections((prev) => ({ ...prev, prescribe: true })) },
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Stethoscope className="h-4 w-4" />
          System Diagnostics
        </CardTitle>
        <CardDescription>Run checks to verify your environment is properly configured</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Setup Check - auto-loaded */}
        <div className="space-y-2">
          <p className="text-sm font-medium">Setup Check</p>
          {setupLoading ? (
            <p className="text-xs text-muted-foreground">Loading prerequisites...</p>
          ) : setup?.checks ? (
            <div className="flex flex-wrap gap-1.5">
              {setup.checks.map((check) => (
                <Badge
                  key={check.name}
                  variant={
                    check.status === 'pass'
                      ? 'success'
                      : check.status === 'warn'
                        ? 'warning'
                        : 'error'
                  }
                  title={check.detail}
                >
                  {check.name}
                </Badge>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">Unable to load setup checks</p>
          )}
        </div>

        {/* Preflight Check */}
        <div className="space-y-2">
          <Button
            variant="outline"
            size="sm"
            onClick={runPreflight}
            disabled={preflight.isPending}
          >
            <Play className="h-3.5 w-3.5" />
            Run Preflight
          </Button>
          {preflight.error && (
            <p className="text-sm text-destructive">{preflight.error.message}</p>
          )}
          {preflight.data && (
            <CollapsibleResult
              label="Preflight Results"
              isOpen={openSections.preflight ?? false}
              onToggle={() => toggle('preflight')}
            >
              <CheckList checks={(preflight.data as { checks?: CheckItem[] }).checks ?? []} />
            </CollapsibleResult>
          )}
        </div>

        {/* Disk Preflight */}
        <div className="space-y-2">
          <Button
            variant="outline"
            size="sm"
            onClick={runDisk}
            disabled={diskPreflight.isPending}
          >
            <HardDrive className="h-3.5 w-3.5" />
            Disk Preflight
          </Button>
          {diskPreflight.error && (
            <p className="text-sm text-destructive">{diskPreflight.error.message}</p>
          )}
          {diskPreflight.data && (
            <CollapsibleResult
              label="Disk Analysis"
              isOpen={openSections.disk ?? false}
              onToggle={() => toggle('disk')}
            >
              <pre className="mt-2 text-xs font-mono text-muted-foreground whitespace-pre-wrap overflow-x-auto">
                {JSON.stringify(diskPreflight.data, null, 2)}
              </pre>
            </CollapsibleResult>
          )}
        </div>

        {/* AI Recommendations */}
        <div className="space-y-2">
          <Button
            variant="outline"
            size="sm"
            onClick={runPrescribe}
            disabled={prescribe.isPending}
          >
            <Sparkles className="h-3.5 w-3.5" />
            AI Recommendations
          </Button>
          {prescribe.error && (
            <p className="text-sm text-destructive">{prescribe.error.message}</p>
          )}
          {prescribe.data && (
            <CollapsibleResult
              label="Recommendations"
              isOpen={openSections.prescribe ?? false}
              onToggle={() => toggle('prescribe')}
            >
              <pre className="mt-2 text-xs font-mono text-muted-foreground whitespace-pre-wrap overflow-x-auto leading-relaxed">
                {typeof (prescribe.data as { recommendations?: string }).recommendations === 'string'
                  ? (prescribe.data as { recommendations?: string }).recommendations
                  : JSON.stringify(prescribe.data, null, 2)}
              </pre>
            </CollapsibleResult>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
