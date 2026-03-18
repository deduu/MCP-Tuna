import { useState } from 'react'
import { Tab, TabList, TabPanel, Tabs } from '@/components/ui/tabs'
import { CompareChatView } from './CompareChatView'
import { SingleChatView } from './SingleChatView'

export function ChatPage() {
  const [activeTab, setActiveTab] = useState('single')

  return (
    <div className="h-[calc(100vh-3.5rem)] -m-6 px-6 py-4">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
        <TabList className="mb-4">
          <Tab value="single">Single Chat</Tab>
          <Tab value="compare">Compare</Tab>
        </TabList>

        <div className="min-h-0 flex-1">
          <TabPanel value="single" className="h-full">
            <SingleChatView />
          </TabPanel>
          <TabPanel value="compare" className="h-full">
            <CompareChatView />
          </TabPanel>
        </div>
      </Tabs>
    </div>
  )
}
