import { Database } from 'lucide-react'
import { useState } from 'react'
import { useDatasetBlendJobs, useDatasetLibrary } from '@/api/hooks/useDatasets'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabList, Tab, TabPanel } from '@/components/ui/tabs'
import { getDatasetLibraryRoots } from '@/lib/dataset-library-roots'
import { DatasetLibrary } from './DatasetLibrary'
import { ImportGenerateTab } from './ImportGenerateTab'
import { CleanNormalizeTab } from './CleanNormalizeTab'
import { EvaluateTab } from './EvaluateTab'
import { DatasetJobTracker } from './DatasetJobTracker'

export function DatasetsPage() {
  const [scanRoots, setScanRoots] = useState<string[]>(() => getDatasetLibraryRoots())
  const { data: datasetLibrary, isLoading, refetch } = useDatasetLibrary(scanRoots)
  const { data: datasetJobs = [] } = useDatasetBlendJobs(12)
  const [activeTab, setActiveTab] = useState('library')
  const datasets = datasetLibrary?.datasets ?? []
  const activeDatasetJobs = datasetJobs.filter((job) => job.status === 'running' || job.status === 'pending').length

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6">
      <div className="flex items-center gap-3">
        <Database className="h-6 w-6 text-primary" />
        <h1 className="text-2xl font-bold">Datasets</h1>
        {datasets && (
          <Badge variant="secondary">{datasets.length} datasets</Badge>
        )}
        <Badge variant="outline">{activeDatasetJobs} active jobs</Badge>
      </div>

      <div className="rounded-lg border border-border/60 bg-secondary/10 p-5 space-y-3">
        <div className="space-y-1">
          <h2 className="text-lg font-semibold">Create, track, then work from the library</h2>
          <p className="text-sm text-muted-foreground">
            Dataset creation can now run in the background. Start blends or document imports in
            Create, watch progress in Dataset Jobs, then continue from Library once outputs land.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">HF blends run as jobs</Badge>
          <Badge variant="outline">Outputs save to disk automatically</Badge>
          <Badge variant="outline">Library refreshes when jobs finish</Badge>
        </div>
      </div>

      <DatasetJobTracker />

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabList>
          <Tab value="library">Library</Tab>
          <Tab value="import">Create</Tab>
          <Tab value="clean">Clean &amp; Normalize</Tab>
          <Tab value="evaluate">Evaluate</Tab>
        </TabList>

        <TabPanel value="library">
          <DatasetLibrary
            datasets={datasets}
            isLoading={isLoading}
            scanRoots={scanRoots}
            prunedStaleRecords={datasetLibrary?.pruned_stale_records ?? 0}
            onRefresh={() => {
              void refetch()
            }}
            onScanRootsChange={setScanRoots}
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
