import { useEffect } from 'react'
import { useLocation } from 'react-router'
import { Settings } from 'lucide-react'
import { GatewayConnection } from './GatewayConnection'
import { DiagnosticsSection } from './DiagnosticsSection'
import { EnvironmentSection } from './EnvironmentSection'
import { ApiKeysSection } from './ApiKeysSection'
import { MaintenanceSection } from './MaintenanceSection'
import { DatasetStorageSection } from './DatasetStorageSection'

export function SettingsPage() {
  const location = useLocation()

  useEffect(() => {
    if (!location.hash) return
    const id = location.hash.replace('#', '')
    const node = document.getElementById(id)
    if (node) {
      node.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [location.hash])

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

      <div id="gateway">
        <GatewayConnection />
      </div>
      <div id="diagnostics">
        <DiagnosticsSection />
      </div>
      <div id="environment">
        <EnvironmentSection />
      </div>
      <div id="storage">
        <DatasetStorageSection />
      </div>
      <div id="providers">
        <ApiKeysSection />
      </div>
      <div id="maintenance">
        <MaintenanceSection />
      </div>
    </div>
  )
}
