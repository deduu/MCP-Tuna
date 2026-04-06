import { AlertTriangle, CheckCircle2 } from 'lucide-react'
import { usePreferenceDatasetAnalysis, useTrainingCapabilities } from '@/api/hooks/useTraining'
import type { PreferenceDatasetAnalysisResult, PreferenceTechnique } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface PreferenceDatasetAnalysisCardProps {
  datasetPath: string
  technique: PreferenceTechnique | null
  className?: string
  onApplyStartingRecipe?: (recipe: Record<string, string | number | boolean>) => void
  applyLabel?: string
}

function buildTechniqueMetrics(result: PreferenceDatasetAnalysisResult): Array<{ label: string; value: string }> {
  if (result.technique_analyzed === 'dpo' && result.dpo) {
    return [
      { label: 'Prompt Unique', value: `${Math.round(result.prompt_stats.unique_ratio * 100)}%` },
      { label: 'Rejected Unique', value: `${Math.round(result.dpo.rejected_stats.unique_ratio * 100)}%` },
      { label: 'Hard Negatives', value: `${Math.round(result.dpo.hard_negative_ratio * 100)}%` },
      { label: 'Dominant Reject', value: `${Math.round(result.dpo.dominant_rejected_ratio * 100)}%` },
    ]
  }

  if (result.technique_analyzed === 'grpo' && result.grpo) {
    return [
      { label: 'Prompt Unique', value: `${Math.round(result.prompt_stats.unique_ratio * 100)}%` },
      { label: 'Resp/Row', value: result.grpo.avg_responses_per_row.toFixed(2) },
      { label: 'Zero Variance', value: `${Math.round(result.grpo.zero_reward_variance_ratio * 100)}%` },
    ]
  }

  if (result.technique_analyzed === 'kto' && result.kto) {
    return [
      { label: 'Prompt Unique', value: `${Math.round(result.prompt_stats.unique_ratio * 100)}%` },
      { label: 'Positive Labels', value: `${Math.round(result.kto.positive_ratio * 100)}%` },
      { label: 'Completion Unique', value: `${Math.round(result.kto.completion_stats.unique_ratio * 100)}%` },
    ]
  }

  return []
}

function buildRepeatedExamples(result: PreferenceDatasetAnalysisResult) {
  if (result.technique_analyzed === 'dpo' && result.dpo) {
    return {
      label: 'Repeated rejected answers',
      items: result.dpo.rejected_stats.top_repeated,
    }
  }

  if (result.technique_analyzed === 'grpo' && result.grpo) {
    return {
      label: 'Repeated candidate responses',
      items: result.grpo.top_repeated_responses,
    }
  }

  if (result.technique_analyzed === 'kto' && result.kto) {
    return {
      label: 'Repeated completions',
      items: result.kto.completion_stats.top_repeated,
    }
  }

  return null
}

function formatRecipeValue(value: string | number | boolean) {
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No'
  }
  return String(value)
}

export function PreferenceDatasetAnalysisCard({
  datasetPath,
  technique,
  className,
  onApplyStartingRecipe,
  applyLabel = 'Apply Safe Defaults',
}: PreferenceDatasetAnalysisCardProps) {
  const normalizedPath = datasetPath.trim()
  const { data: capabilities } = useTrainingCapabilities()
  const supported = capabilities?.supports_preference_dataset_analysis ?? false
  const analysis = usePreferenceDatasetAnalysis(normalizedPath, technique, supported)

  if (!normalizedPath || !technique || !supported) {
    return null
  }

  if (analysis.isLoading) {
    return (
      <Card className={cn('border-dashed', className)}>
        <CardContent className="p-4 text-sm text-muted-foreground">
          Analyzing {technique.toUpperCase()} dataset quality...
        </CardContent>
      </Card>
    )
  }

  if (analysis.isError || !analysis.data) {
    return (
      <Card className={cn('border-dashed', className)}>
        <CardContent className="p-4 text-sm text-muted-foreground">
          Preference dataset analysis is unavailable for this path right now.
        </CardContent>
      </Card>
    )
  }

  const result = analysis.data
  const metrics = buildTechniqueMetrics(result)
  const repeated = buildRepeatedExamples(result)
  const isWarn = result.status === 'warn'
  const guidance = result.guidance

  return (
    <Card className={cn(isWarn ? 'border-amber-500/40 bg-amber-50/40' : 'border-emerald-500/30 bg-emerald-50/30', className)}>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          {isWarn ? <AlertTriangle className="h-4 w-4 text-amber-600" /> : <CheckCircle2 className="h-4 w-4 text-emerald-600" />}
          Preference Dataset Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Badge variant={isWarn ? 'secondary' : 'outline'}>
            {result.technique_analyzed.toUpperCase()}
          </Badge>
          <Badge variant="outline">
            {result.analyzed_row_count}/{result.row_count} rows analyzed
          </Badge>
          <Badge variant="outline">
            risk: {result.risk_level}
          </Badge>
          {result.truncated && (
            <Badge variant="outline">
              sampled first rows
            </Badge>
          )}
          {guidance && onApplyStartingRecipe && Object.keys(guidance.starting_recipe).length > 0 && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="ml-auto h-7 text-[11px]"
              onClick={() => onApplyStartingRecipe(guidance.starting_recipe)}
            >
              {applyLabel}
            </Button>
          )}
        </div>

        {metrics.length > 0 && (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {metrics.map((metric) => (
              <div key={metric.label} className="rounded-md border border-border/60 bg-background/70 px-3 py-2">
                <div className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">{metric.label}</div>
                <div className="mt-1 text-sm font-medium">{metric.value}</div>
              </div>
            ))}
          </div>
        )}

        {result.warnings.length > 0 && (
          <div className="space-y-1">
            <p className="text-xs font-medium text-foreground">Warnings</p>
            {result.warnings.slice(0, 3).map((warning) => (
              <p key={warning} className="text-xs text-muted-foreground">
                {warning}
              </p>
            ))}
          </div>
        )}

        {result.recommendations.length > 0 && (
          <div className="space-y-1">
            <p className="text-xs font-medium text-foreground">Recommended fixes</p>
            {result.recommendations.slice(0, 2).map((recommendation) => (
              <p key={recommendation} className="text-xs text-muted-foreground">
                {recommendation}
              </p>
            ))}
          </div>
        )}

        {guidance && (
          <div className="space-y-3 rounded-md border border-border/60 bg-background/60 p-3">
            <div className="space-y-1">
              <p className="text-xs font-medium text-foreground">What MCP-Tuna would watch for</p>
              <p className="text-xs text-muted-foreground">{guidance.headline}</p>
            </div>

            {Object.keys(guidance.starting_recipe).length > 0 && (
              <div className="grid gap-2 sm:grid-cols-2">
                {Object.entries(guidance.starting_recipe).map(([key, value]) => (
                  <div key={key} className="rounded-md border border-border/60 bg-background/70 px-3 py-2">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div className="mt-1 text-sm font-medium">{formatRecipeValue(value)}</div>
                  </div>
                ))}
              </div>
            )}

            {guidance.recommended_actions.length > 0 && (
              <div className="space-y-1">
                <p className="text-xs font-medium text-foreground">Suggested next steps</p>
                {guidance.recommended_actions.slice(0, 3).map((action) => (
                  <p key={action} className="text-xs text-muted-foreground">
                    {action}
                  </p>
                ))}
              </div>
            )}

            {guidance.hidden_factors.length > 0 && (
              <div className="space-y-1">
                <p className="text-xs font-medium text-foreground">Things users often miss</p>
                {guidance.hidden_factors.slice(0, 3).map((item) => (
                  <p key={item} className="text-xs text-muted-foreground">
                    {item}
                  </p>
                ))}
              </div>
            )}
          </div>
        )}

        {repeated && repeated.items.length > 0 && (
          <div className="space-y-1">
            <p className="text-xs font-medium text-foreground">{repeated.label}</p>
            {repeated.items.slice(0, 2).map((item) => (
              <p key={`${item.preview}-${item.count}`} className="text-xs text-muted-foreground">
                {item.preview} x{item.count}
              </p>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
