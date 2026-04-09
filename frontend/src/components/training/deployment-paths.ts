import type { ThinkingMode, TrainingJob } from '@/api/types'
import type { DeployDialogInitialValues } from '@/components/deployments/DeployDialog'

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object'
}

function basename(value: string): string {
  const normalized = value.trim().replace(/\\/g, '/')
  const leaf = normalized.split('/').pop() || normalized
  return leaf.replace(/\.[^.]+$/, '')
}

function readTrimmedString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function readThinkingMode(value: unknown): ThinkingMode | undefined {
  return value === 'default' || value === 'on' || value === 'off'
    ? value
    : undefined
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

function resolveDeploymentBaseModel(result: unknown): string | null {
  if (!isRecord(result)) return null

  const adapterBaseModel = readTrimmedString(result.adapter_base_model)
  if (adapterBaseModel) return adapterBaseModel

  const stageResults = result.stage_results
  if (Array.isArray(stageResults) && stageResults.length > 0) {
    const lastStage = stageResults[stageResults.length - 1]
    if (isRecord(lastStage) && isRecord(lastStage.training_result)) {
      const nestedBaseModel = resolveDeploymentBaseModel(lastStage.training_result)
      if (nestedBaseModel) return nestedBaseModel
    }
  }

  return readTrimmedString(result.base_model)
}

function resolveDeploymentThinkingMode(result: unknown): ThinkingMode | undefined {
  if (!isRecord(result)) return undefined

  const config = isRecord(result.config) ? result.config : null
  const directThinkingMode = readThinkingMode(config?.thinking_mode)
  if (directThinkingMode) return directThinkingMode

  const stageResults = result.stage_results
  if (Array.isArray(stageResults) && stageResults.length > 0) {
    const lastStage = stageResults[stageResults.length - 1]
    if (isRecord(lastStage) && isRecord(lastStage.training_result)) {
      return resolveDeploymentThinkingMode(lastStage.training_result)
    }
  }

  return undefined
}

export function getDeployInitialValues(job: TrainingJob): DeployDialogInitialValues | null {
  const outputPath = getTrainingOutputPath(job).trim()
  if (!outputPath) return null
  const config = isRecord(job.result) && isRecord(job.result.config) ? job.result.config : null
  const modality = config?.trainer === 'vlm_sft' ? 'vision-language' : 'text'
  const thinkingMode = modality === 'text' ? resolveDeploymentThinkingMode(job.result) : undefined

  if (trainingUsesAdapter(job.result)) {
    const resolvedBaseModel = resolveDeploymentBaseModel(job.result) ?? job.base_model.trim()
    const baseName = basename(resolvedBaseModel)
    const adapterName = basename(outputPath)
    return {
      name: [baseName, adapterName].filter(Boolean).join(' + '),
      modelPath: resolvedBaseModel,
      adapterPath: outputPath,
      modality,
      thinkingMode,
    }
  }

  return {
    name: basename(outputPath),
    modelPath: outputPath,
    modality,
    thinkingMode,
  }
}
