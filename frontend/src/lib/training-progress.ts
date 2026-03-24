interface TrainingLogEntryLike {
  step?: number | null
  loss?: number | null
  eval_loss?: number | null
  learning_rate?: number | null
}

interface TrainingProgressLike {
  max_steps?: number | null
  log_history?: TrainingLogEntryLike[] | null
}

export interface LossChartPoint {
  step: number
  loss?: number
  evalLoss?: number
  learning_rate?: number
}

function asFiniteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

export function isTrainingStageName(stage?: string | null): boolean {
  return stage === 'train' || stage?.startsWith('finetune.train') === true
}

export function includesTrainingStage(steps?: string[] | null): boolean {
  return (steps ?? []).some((step) => isTrainingStageName(step))
}

export function hasTrainerProgress(progress?: TrainingProgressLike | null): boolean {
  return (progress?.max_steps ?? 0) > 0
}

export function isTrainingStageActive(
  stage?: string | null,
  progress?: TrainingProgressLike | null,
): boolean {
  return isTrainingStageName(stage) && hasTrainerProgress(progress)
}

export function buildLossChartData(progress?: TrainingProgressLike | null): LossChartPoint[] {
  return (
    progress?.log_history?.flatMap((entry) => {
      const step = asFiniteNumber(entry.step)
      if (step == null) {
        return []
      }

      const loss = asFiniteNumber(entry.loss)
      const evalLoss = asFiniteNumber(entry.eval_loss)
      if (loss == null && evalLoss == null) {
        return []
      }

      return [{
        step,
        loss,
        evalLoss,
        learning_rate: asFiniteNumber(entry.learning_rate),
      }]
    }) ?? []
  )
}
