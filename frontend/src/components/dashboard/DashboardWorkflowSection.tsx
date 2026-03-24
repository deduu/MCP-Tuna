import type { LucideIcon } from 'lucide-react'
import {
  ArrowRight,
  BarChart3,
  Database,
  GitBranch,
  MessageSquare,
  Rocket,
  Sparkles,
  Wrench,
} from 'lucide-react'
import { useNavigate } from 'react-router'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface WorkflowGroup {
  title: string
  description: string
  color: string
  actions: Array<{
    label: string
    description: string
    path: string
    icon: LucideIcon
  }>
}

const WORKFLOW_GROUPS: WorkflowGroup[] = [
  {
    title: 'Prepare',
    description: 'Import, generate, and clean the data you will actually train on.',
    color: 'var(--color-ns-generate)',
    actions: [
      {
        label: 'Dataset Library',
        description: 'Browse existing files, inspect counts, and pick reusable inputs.',
        path: '/datasets',
        icon: Database,
      },
      {
        label: 'Pipeline Builder',
        description: 'Run guided end-to-end workflows when you want fewer manual steps.',
        path: '/pipeline',
        icon: GitBranch,
      },
    ],
  },
  {
    title: 'Build',
    description: 'Start fine-tuning jobs with the fewest clicks from the current state.',
    color: 'var(--color-ns-finetune)',
    actions: [
      {
        label: 'Training Jobs',
        description: 'Launch, monitor, and resume supervised or preference optimization runs.',
        path: '/training',
        icon: Sparkles,
      },
      {
        label: 'Agent Chat',
        description: 'Use the agent when you need tool-assisted exploration or experimentation.',
        path: '/chat',
        icon: MessageSquare,
      },
    ],
  },
  {
    title: 'Ship',
    description: 'Deploy, validate, and inspect tools from the same operational flow.',
    color: 'var(--color-ns-host)',
    actions: [
      {
        label: 'Deployments',
        description: 'Turn trained outputs into MCP servers or REST APIs.',
        path: '/deployments',
        icon: Rocket,
      },
      {
        label: 'Evaluation',
        description: 'Run judges, benchmarks, and comparisons before or after release.',
        path: '/evaluation',
        icon: BarChart3,
      },
      {
        label: 'Tool Explorer',
        description: 'Inspect available tools when you need lower-level execution control.',
        path: '/tools',
        icon: Wrench,
      },
    ],
  },
]

export function DashboardWorkflowSection() {
  const navigate = useNavigate()

  return (
    <Card className="border-border/70">
      <CardHeader>
        <CardTitle className="text-base">Start A Workflow</CardTitle>
        <p className="text-sm text-muted-foreground">
          Grouped by the jobs users actually do, not by isolated screens.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {WORKFLOW_GROUPS.map((group) => (
          <div key={group.title} className="rounded-2xl border border-border/70 bg-secondary/10 p-4">
            <div className="mb-4 flex items-start gap-3">
              <span
                className="mt-0.5 h-3 w-3 rounded-full"
                style={{ backgroundColor: group.color }}
              />
              <div>
                <h4 className="text-sm font-semibold">{group.title}</h4>
                <p className="mt-1 text-sm text-muted-foreground">{group.description}</p>
              </div>
            </div>

            <div className="space-y-2">
              {group.actions.map((action) => (
                <button
                  key={action.path}
                  type="button"
                  onClick={() => navigate(action.path)}
                  className="flex w-full items-start gap-3 rounded-xl border border-border/70 bg-background/40 p-3 text-left transition-colors hover:border-primary/35 hover:bg-background/70"
                >
                  <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-border/60 bg-secondary/40">
                    <action.icon className="h-4 w-4 text-primary" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-medium">{action.label}</p>
                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">{action.description}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
