import { useNavigate } from 'react-router'
import { Card, CardContent } from '@/components/ui/card'
import { Sparkles, FlaskConical, Rocket, GitBranch, MessageSquare, BarChart3 } from 'lucide-react'

const ACTIONS = [
  {
    label: 'Generate Data',
    description: 'Create SFT/DPO/GRPO datasets from documents',
    icon: Sparkles,
    path: '/datasets',
    color: 'var(--color-ns-generate)',
  },
  {
    label: 'Train Model',
    description: 'Fine-tune with LoRA/QLoRA',
    icon: FlaskConical,
    path: '/training',
    color: 'var(--color-ns-finetune)',
  },
  {
    label: 'Deploy Model',
    description: 'Host as MCP server or REST API',
    icon: Rocket,
    path: '/deployments',
    color: 'var(--color-ns-host)',
  },
  {
    label: 'Run Pipeline',
    description: 'End-to-end workflow automation',
    icon: GitBranch,
    path: '/pipeline',
    color: 'var(--color-ns-workflow)',
  },
  {
    label: 'Chat with Agent',
    description: 'Interactive agent with tool access',
    icon: MessageSquare,
    path: '/chat',
    color: 'var(--color-primary)',
  },
  {
    label: 'Evaluate',
    description: 'LLM judge, fine-tune eval & benchmarks',
    icon: BarChart3,
    path: '/evaluation',
    color: 'var(--color-ns-evaluate)',
  },
]

export function QuickActions() {
  const navigate = useNavigate()

  return (
    <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
      {ACTIONS.map((action) => (
        <Card
          key={action.path}
          className="cursor-pointer hover:border-primary/30 transition-colors group"
          onClick={() => navigate(action.path)}
        >
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <div
                className="p-2 rounded-lg"
                style={{ backgroundColor: `color-mix(in srgb, ${action.color} 15%, transparent)` }}
              >
                <action.icon className="h-4 w-4" style={{ color: action.color }} />
              </div>
              <div>
                <p className="text-sm font-medium group-hover:text-primary transition-colors">
                  {action.label}
                </p>
                <p className="text-xs text-muted-foreground mt-0.5">{action.description}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
