import { GitBranch } from 'lucide-react'
import { Tabs, TabList, Tab, TabPanel } from '@/components/ui/tabs'
import { PipelineTemplates } from './PipelineTemplates'
import { PipelineJobTracker } from './PipelineJobTracker'
import { OrchestrationTab } from './OrchestrationTab'

export function PipelinePage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <GitBranch className="h-6 w-6 text-primary" />
        <h1 className="text-2xl font-bold tracking-tight">Pipeline Builder</h1>
      </div>

      <Tabs defaultValue="pipelines">
        <TabList>
          <Tab value="pipelines">Pipelines</Tab>
          <Tab value="orchestration">Orchestration</Tab>
        </TabList>

        <TabPanel value="pipelines">
          <div className="space-y-8">
            <PipelineTemplates />
            <PipelineJobTracker />
          </div>
        </TabPanel>

        <TabPanel value="orchestration">
          <OrchestrationTab />
        </TabPanel>
      </Tabs>
    </div>
  )
}
