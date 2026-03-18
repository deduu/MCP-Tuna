import type { TrainingJob } from '@/api/types'
import type { DeployDialogInitialValues } from '@/components/deployments/DeployDialog'

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object'
}

export function getTrainingOutputPath(job: TrainingJob): string {
  if (isRecord(job.result)) {
    const modelPath = job.result.model_path
    if (typeof modelPath === 'string' && modelPath.trim()) return modelPath

    const finalModelPath = job.result.final_model_path
    if (typeof finalModelPath === 'string' && finalModelPath.trim()) return finalModelPath
  }

  return job.output_dir
}

export function trainingUsesAdapter(result: unknown): boolean {
  if (!isRecord(result)) return true

  const config = isRecord(result.config) ? result.config : null
  if (config) {
    if (typeof config.use_lora === 'boolean') return config.use_lora
    if (config.trainer === 'grpo') return false
  }

  const stageResults = result.stage_results
  if (Array.isArray(stageResults) && stageResults.length > 0) {
    const lastStage = stageResults[stageResults.length - 1]
    if (isRecord(lastStage) && isRecord(lastStage.training_result)) {
      return trainingUsesAdapter(lastStage.training_result)
    }
  }

  return true
}

export function getDeployInitialValues(job: TrainingJob): DeployDialogInitialValues | null {
  const outputPath = getTrainingOutputPath(job).trim()
  if (!outputPath) return null
  const config = isRecord(job.result) && isRecord(job.result.config) ? job.result.config : null
  const modality = config?.trainer === 'vlm_sft' ? 'vision-language' : 'text'

  if (trainingUsesAdapter(job.result)) {
    return {
      modelPath: job.base_model,
      adapterPath: outputPath,
      modality,
    }
  }

  return {
    modelPath: outputPath,
    modality,
  }
}
