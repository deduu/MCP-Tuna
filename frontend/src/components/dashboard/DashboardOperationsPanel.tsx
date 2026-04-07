import { ShieldCheck, Sparkles } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Tab, TabList, TabPanel, Tabs } from '@/components/ui/tabs'
import {
  DashboardReadinessPanel,
  type DashboardReadinessPanelProps,
} from './DashboardReadinessPanel'
import { DashboardWorkflowSection } from './DashboardWorkflowSection'

type DashboardOperationsPanelProps = Omit<DashboardReadinessPanelProps, 'cardless'>

export function DashboardOperationsPanel(props: DashboardOperationsPanelProps) {
  return (
    <Tabs defaultValue="workflow">
      <Card className="border-border/70">
        <div className="px-6 pt-6">
          <TabList className="mb-0 gap-6">
            <Tab value="workflow" className="flex items-center gap-2 px-0">
              <Sparkles className="h-4 w-4" />
              Workflows
            </Tab>
            <Tab value="readiness" className="flex items-center gap-2 px-0">
              <ShieldCheck className="h-4 w-4" />
              Readiness
            </Tab>
          </TabList>
        </div>
        <CardContent className="pt-6">
          <TabPanel value="workflow">
            <DashboardWorkflowSection cardless />
          </TabPanel>
          <TabPanel value="readiness">
            <DashboardReadinessPanel {...props} cardless />
          </TabPanel>
        </CardContent>
      </Card>
    </Tabs>
  )
}
