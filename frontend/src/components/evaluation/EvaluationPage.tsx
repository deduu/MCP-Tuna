import { BarChart3 } from 'lucide-react'
import { useState } from 'react'
import { Tabs, TabList, Tab, TabPanel } from '@/components/ui/tabs'
import { JudgeTab } from './JudgeTab'
import { FtEvalTab } from './FtEvalTab'
import { BenchmarkTab } from './BenchmarkTab'

export function EvaluationPage() {
  const [activeTab, setActiveTab] = useState('judge')

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6">
      <div className="flex items-center gap-3">
        <BarChart3 className="h-6 w-6 text-primary" />
        <h1 className="text-2xl font-bold">Evaluation Hub</h1>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabList>
          <Tab value="judge">LLM Judge</Tab>
          <Tab value="ft-eval">Fine-tune Eval</Tab>
          <Tab value="benchmark">Model Benchmark</Tab>
        </TabList>

        <TabPanel value="judge">
          <JudgeTab />
        </TabPanel>

        <TabPanel value="ft-eval">
          <FtEvalTab />
        </TabPanel>

        <TabPanel value="benchmark">
          <BenchmarkTab />
        </TabPanel>
      </Tabs>
    </div>
  )
}
