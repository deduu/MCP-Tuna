import { Settings } from 'lucide-react'
import { GatewayConnection } from './GatewayConnection'
import { DiagnosticsSection } from './DiagnosticsSection'
import { EnvironmentSection } from './EnvironmentSection'
import { MaintenanceSection } from './MaintenanceSection'

export function SettingsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Settings className="h-6 w-6 text-muted-foreground" />
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
          <p className="text-sm text-muted-foreground">
            Gateway connection, diagnostics, and system configuration
          </p>
        </div>
      </div>

      <GatewayConnection />
      <DiagnosticsSection />
      <EnvironmentSection />
      <MaintenanceSection />
    </div>
  )
}
