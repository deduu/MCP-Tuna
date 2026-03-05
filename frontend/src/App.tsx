import { BrowserRouter, Routes, Route } from 'react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import { AppShell } from '@/components/layout/AppShell'
import { DashboardPage } from '@/components/dashboard/DashboardPage'
import { ToolExplorerPage } from '@/components/tools/ToolExplorerPage'
import { NamespaceDetailPage } from '@/components/tools/NamespaceDetailPage'
import { ToolExecutionPage } from '@/components/tools/ToolExecutionPage'
import { ChatPage } from '@/components/chat/ChatPage'
import { PlaceholderPage } from '@/components/placeholder/PlaceholderPage'

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
            <Route
              path="pipeline"
              element={<PlaceholderPage title="Pipeline Builder" description="Visual node-based workflow builder with React Flow. Coming in Phase 5." />}
            />
            <Route
              path="datasets"
              element={<PlaceholderPage title="Dataset Manager" description="Upload, preview, split, merge and manage datasets. Coming in Phase 4." />}
            />
            <Route
              path="training"
              element={<PlaceholderPage title="Training Jobs" description="Real-time training monitoring with loss curves and GPU stats. Coming in Phase 3." />}
            />
            <Route
              path="deployments"
              element={<PlaceholderPage title="Deployments" description="Model deployment management with health monitoring. Coming in Phase 6." />}
            />
            <Route
              path="evaluation"
              element={<PlaceholderPage title="Evaluation Hub" description="LLM-as-a-judge, fine-tune evaluation, and model benchmarking. Coming in Phase 6." />}
            />
            <Route
              path="settings"
              element={<PlaceholderPage title="Settings" description="API keys, gateway URL, and theme configuration. Coming in Phase 7." />}
            />
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
