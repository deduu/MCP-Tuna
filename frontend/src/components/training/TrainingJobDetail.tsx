import type { TrainingJob } from '@/api/types'
import { Progress } from '@/components/ui/progress'
import { Card, CardContent } from '@/components/ui/card'
import { LossChart } from './LossChart'

interface TrainingJobDetailProps {
  job: TrainingJob
}

export function TrainingJobDetail({ job }: TrainingJobDetailProps) {
  const p = job.progress

  const chartData =
    p?.log_history?.map((entry) => ({
      step: entry.step,
      loss: entry.loss,
      learning_rate: entry.learning_rate,
    })) ?? []

  return (
    <div className="space-y-4 pt-3 border-t border-border">
      {/* Loss chart */}
      <LossChart data={chartData} />

      {/* Stats grid */}
      {p && (
        <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Step</span>
            <span className="font-mono">
              {p.current_step} / {p.max_steps}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Epoch</span>
            <span className="font-mono">
              {typeof p.current_epoch === 'number' ? p.current_epoch.toFixed(1) : '?'} / {p.max_epochs}
            </span>
          </div>
          {typeof p.learning_rate === 'number' && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">Learning Rate</span>
              <span className="font-mono">{p.learning_rate.toExponential(2)}</span>
            </div>
          )}
          {p.grad_norm != null && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">Grad Norm</span>
              <span className="font-mono">{p.grad_norm.toFixed(4)}</span>
            </div>
          )}
          {p.eval_loss != null && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">Eval Loss</span>
              <span className="font-mono">{p.eval_loss.toFixed(4)}</span>
            </div>
          )}
        </div>
      )}

      {/* GPU memory bar */}
      {p?.gpu_memory_used_gb != null && p.gpu_memory_total_gb != null && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>GPU Memory</span>
            <span>
              {p.gpu_memory_used_gb.toFixed(1)} / {p.gpu_memory_total_gb.toFixed(1)} GB
            </span>
          </div>
          <Progress
            value={p.gpu_memory_used_gb}
            max={p.gpu_memory_total_gb}
            color="var(--color-ns-finetune, var(--color-primary))"
          />
        </div>
      )}

      {/* Error message */}
      {job.status === 'failed' && job.error && (
        <Card className="border-destructive bg-destructive/5">
          <CardContent className="p-3 text-sm text-red-400">
            <p className="font-medium mb-1">Error</p>
            <p className="font-mono text-xs whitespace-pre-wrap">{job.error}</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
