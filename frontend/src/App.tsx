import { BrowserRouter, Routes, Route } from 'react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import { AppShell } from '@/components/layout/AppShell'
import { DashboardPage } from '@/components/dashboard/DashboardPage'
import { ToolExplorerPage } from '@/components/tools/ToolExplorerPage'
import { NamespaceDetailPage } from '@/components/tools/NamespaceDetailPage'
import { ToolExecutionPage } from '@/components/tools/ToolExecutionPage'
import { ChatPage } from '@/components/chat/ChatPage'
import { DatasetsPage } from '@/components/datasets/DatasetsPage'
import { TrainingPage } from '@/components/training/TrainingPage'
import { DeploymentsPage } from '@/components/deployments/DeploymentsPage'
import { EvaluationPage } from '@/components/evaluation/EvaluationPage'
import { PipelinePage } from '@/components/pipeline/PipelinePage'
import { SettingsPage } from '@/components/settings/SettingsPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<DashboardPage />} />
            <Route path="tools" element={<ToolExplorerPage />} />
            <Route path="tools/:namespace" element={<NamespaceDetailPage />} />
            <Route path="tools/:namespace/:tool" element={<ToolExecutionPage />} />
            <Route path="chat" element={<ChatPage />} />
            <Route path="pipeline" element={<PipelinePage />} />
            <Route path="datasets" element={<DatasetsPage />} />
            <Route path="training" element={<TrainingPage />} />
            <Route path="deployments" element={<DeploymentsPage />} />
            <Route path="evaluation" element={<EvaluationPage />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster
        theme="dark"
        position="bottom-right"
        toastOptions={{
          style: {
            background: 'var(--color-card)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-foreground)',
          },
        }}
      />
    </QueryClientProvider>
  )
}

export default App
