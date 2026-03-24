import { Database } from 'lucide-react'
import { useState } from 'react'
import { useDatasets } from '@/api/hooks/useDatasets'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabList, Tab, TabPanel } from '@/components/ui/tabs'
import { DatasetLibrary } from './DatasetLibrary'
import { ImportGenerateTab } from './ImportGenerateTab'
import { CleanNormalizeTab } from './CleanNormalizeTab'
import { EvaluateTab } from './EvaluateTab'

export function DatasetsPage() {
  const { data: datasets, isLoading } = useDatasets()
  const [activeTab, setActiveTab] = useState('library')

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6">
      <div className="flex items-center gap-3">
        <Database className="h-6 w-6 text-primary" />
        <h1 className="text-2xl font-bold">Datasets</h1>
        {datasets && (
          <Badge variant="secondary">{datasets.length} datasets</Badge>
        )}
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabList>
          <Tab value="library">Library</Tab>
          <Tab value="import">Import &amp; Generate</Tab>
          <Tab value="clean">Clean &amp; Normalize</Tab>
          <Tab value="evaluate">Evaluate</Tab>
        </TabList>

        <TabPanel value="library">
          <DatasetLibrary
            datasets={datasets ?? []}
            isLoading={isLoading}
            onSwitchToImport={() => setActiveTab('import')}
          />
        </TabPanel>

        <TabPanel value="import">
          <ImportGenerateTab />
        </TabPanel>

        <TabPanel value="clean">
          <CleanNormalizeTab />
        </TabPanel>

        <TabPanel value="evaluate">
          <EvaluateTab />
        </TabPanel>
      </Tabs>
    </div>
  )
}
