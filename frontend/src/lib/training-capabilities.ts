import type {
  LocalModelCandidate,
  ModelModality,
  PreferenceTechnique,
  TrainingCapabilitySummary,
  TrainingTechnique,
} from '@/api/types'
import { compactSourceHint, compactTrainingPrefix } from '@/lib/output-naming'

export type DifficultyOrder = 'easy_first' | 'hard_first'
export type QuantizationOption = '4bit' | 'none'

export const DEFAULT_NUM_EPOCHS = 3
export const DEFAULT_LEARNING_RATE = 2e-4
export const DEFAULT_LORA_R = 16
export const DEFAULT_LORA_ALPHA = 32
export const DEFAULT_LORA_DROPOUT = 0.05
export const DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1
export const DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
export const PREFERENCE_TECHNIQUES: PreferenceTechnique[] = ['dpo', 'grpo', 'kto']

export interface PreferenceStartingRecipePatch {
  num_epochs?: number
  learning_rate?: number
  start_from_sft_checkpoint?: boolean
}

export interface TrainingTechniqueOption {
  value: TrainingTechnique
  label: string
  description: string
  enabled: boolean
  reason?: string
}

function buildTimestampedOutputDir(prefix: string, sourcePath?: string): string {
  const now = new Date()
  const stamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    '_',
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0'),
  ].join('')
  const compactPrefix = compactTrainingPrefix(prefix, 18)
  const sourceSuffix = sourcePath ? compactSourceHint(sourcePath, 18, 'src') : ''

  return `./output/${compactPrefix}${sourceSuffix ? `_${sourceSuffix}` : ''}_${stamp}`
}

function resolvePathHint(value: unknown): string {
  if (typeof value === 'string') return value.trim()

  if (Array.isArray(value)) {
    const paths = value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
    if (paths.length === 1) return paths[0]
    if (paths.length > 1) return `multi_${paths.length}_files`
  }

  return ''
}

function parseSequentialStages(stages: unknown): Array<Record<string, unknown>> {
  if (Array.isArray(stages)) {
    return stages.filter((stage): stage is Record<string, unknown> => !!stage && typeof stage === 'object')
  }

  if (typeof stages !== 'string' || !stages.trim()) {
    return []
  }

  try {
    const parsed = JSON.parse(stages)
    return Array.isArray(parsed)
      ? parsed.filter((stage): stage is Record<string, unknown> => !!stage && typeof stage === 'object')
      : []
  } catch {
    return []
  }
}

const VLM_MODEL_MARKERS = [
  'qwen2.5-vl',
  'qwen-vl',
  'llava',
  'llava-next',
  'internvl',
  'idefics',
  'paligemma',
  'phi-3-vision',
  'phi3-vision',
  'minicpm-v',
  'cogvlm',
  'molmo',
  'deepseek-vl',
]

const TEXT_TECHNIQUE_OPTIONS: TrainingTechniqueOption[] = [
  {
    value: 'sft',
    label: 'SFT',
    description: 'Supervised fine-tuning for text instruction datasets.',
    enabled: true,
  },
  {
    value: 'dpo',
    label: 'DPO',
    description: 'Preference optimization using prompt/chosen/rejected data.',
    enabled: true,
  },
  {
    value: 'grpo',
    label: 'GRPO',
    description: 'Reward-optimized text training with prompt/reward tables.',
    enabled: true,
  },
  {
    value: 'kto',
    label: 'KTO',
    description: 'Binary preference training with prompt/completion labels.',
    enabled: true,
  },
  {
    value: 'curriculum',
    label: 'Curriculum',
    description: 'Stage text SFT examples from easy to hard.',
    enabled: true,
  },
  {
    value: 'vlm_sft',
    label: 'VLM SFT',
    description: 'Supervised fine-tuning for multimodal instruction datasets.',
    enabled: false,
    reason: 'Select a vision-language model to review multimodal training support.',
  },
]

const TRAINING_TOOL_NAMES: Partial<Record<TrainingTechnique, { sync: string; async: string }>> = {
  sft: { sync: 'finetune.train', async: 'finetune.train_async' },
  dpo: { sync: 'finetune.train_dpo', async: 'finetune.train_dpo_async' },
  grpo: { sync: 'finetune.train_grpo', async: 'finetune.train_grpo_async' },
  kto: { sync: 'finetune.train_kto', async: 'finetune.train_kto_async' },
  curriculum: { sync: 'finetune.train_curriculum', async: 'finetune.train_curriculum_async' },
  vlm_sft: { sync: 'finetune.train_vlm', async: 'finetune.train_vlm_async' },
  sequential: { sync: 'finetune.sequential_train', async: 'finetune.sequential_train_async' },
}

export function resolveTrainingToolName(
  technique: TrainingTechnique,
  asyncMode: boolean,
): string | null {
  const mapping = TRAINING_TOOL_NAMES[technique]
  if (!mapping) return null
  return asyncMode ? mapping.async : mapping.sync
}

export function supportsAdapterInitialization(technique: TrainingTechnique): boolean {
  return technique === 'sft' || technique === 'dpo' || technique === 'grpo' || technique === 'kto'
}

export function isPreferenceTechnique(value: string | null | undefined): value is PreferenceTechnique {
  return value === 'dpo' || value === 'grpo' || value === 'kto'
}

function parseRecipeNumber(value: string | number | boolean | undefined): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : undefined
  }
  return undefined
}

export function extractPreferenceStartingRecipePatch(
  recipe?: Record<string, string | number | boolean>,
): PreferenceStartingRecipePatch {
  if (!recipe) return {}

  const patch: PreferenceStartingRecipePatch = {}
  const epochs = parseRecipeNumber(recipe.epochs)
  const learningRate = parseRecipeNumber(recipe.learning_rate)

  if (epochs !== undefined) {
    patch.num_epochs = epochs
  }
  if (learningRate !== undefined) {
    patch.learning_rate = learningRate
  }
  if (recipe.start_from_sft_checkpoint === true) {
    patch.start_from_sft_checkpoint = true
  }

  return patch
}

export function resolvePreferenceTechniqueFromTool(
  toolName: string,
  explicitTechnique?: unknown,
): PreferenceTechnique | null {
  if (typeof explicitTechnique === 'string' && isPreferenceTechnique(explicitTechnique)) {
    return explicitTechnique
  }

  if (toolName.includes('train_dpo') || toolName.includes('benchmark_dpo')) return 'dpo'
  if (toolName.includes('train_grpo') || toolName.includes('benchmark_grpo')) return 'grpo'
  if (toolName.includes('train_kto')) return 'kto'
  return null
}

export function defaultsToLoraTraining(technique: TrainingTechnique): boolean {
  return technique !== 'vlm_sft'
}

export function buildDefaultOutputDir(
  technique: TrainingTechnique,
  sequential: boolean,
  sourcePath?: string,
): string {
  const prefix = sequential ? `sequential_${technique}` : technique
  return buildTimestampedOutputDir(prefix, sourcePath)
}

export function buildPipelineOutputDir(
  technique: string,
  sourcePath?: string | string[],
): string {
  return buildTimestampedOutputDir(`pipeline_${technique}`, resolvePathHint(sourcePath))
}

export function buildToolExecutionOutputDir(
  toolName: string,
  values: Record<string, unknown>,
): string | null {
  const datasetPath = resolvePathHint(values.dataset_path)
  const filePath = resolvePathHint(values.file_path)
  const filePaths = resolvePathHint(values.file_paths)

  switch (toolName) {
    case 'finetune.train':
    case 'finetune.train_async':
      return buildDefaultOutputDir('sft', false, datasetPath)
    case 'finetune.train_dpo':
    case 'finetune.train_dpo_async':
      return buildDefaultOutputDir('dpo', false, datasetPath)
    case 'finetune.train_grpo':
    case 'finetune.train_grpo_async':
      return buildDefaultOutputDir('grpo', false, datasetPath)
    case 'finetune.train_kto':
    case 'finetune.train_kto_async':
      return buildDefaultOutputDir('kto', false, datasetPath)
    case 'finetune.train_curriculum':
    case 'finetune.train_curriculum_async':
      return buildDefaultOutputDir('curriculum', false, datasetPath)
    case 'finetune.train_vlm':
    case 'finetune.train_vlm_async':
      return buildDefaultOutputDir('vlm_sft', false, datasetPath)
    case 'finetune.sequential_train':
    case 'finetune.sequential_train_async': {
      const stages = parseSequentialStages(values.stages)
      const firstStage = stages[0]
      const technique = typeof firstStage?.technique === 'string'
        ? firstStage.technique
        : 'sft'
      const sequentialSource = resolvePathHint(firstStage?.dataset_path)
      return buildDefaultOutputDir(technique as TrainingTechnique, true, sequentialSource)
    }
    case 'workflow.full_pipeline':
    case 'workflow.full_pipeline_async': {
      const technique = typeof values.technique === 'string' ? values.technique : 'sft'
      return buildPipelineOutputDir(technique, filePaths || filePath)
    }
    case 'workflow.curriculum_pipeline': {
      const technique = typeof values.technique === 'string' ? values.technique : 'sft'
      return buildPipelineOutputDir(`curriculum_${technique}`, filePaths || filePath)
    }
    default:
      return null
  }
}

export function inferModelModality(
  modelPath: string,
  candidate?: Partial<Pick<LocalModelCandidate, 'id' | 'model_path' | 'usable_for' | 'modality'>>,
): ModelModality {
  if (candidate?.modality) return candidate.modality

  const usableFor = (candidate?.usable_for ?? []).map((value) => value.toLowerCase())
  if (usableFor.some((value) => value.includes('vision') || value.includes('vlm'))) {
    return 'vision-language'
  }

  const haystack = `${candidate?.id ?? ''} ${candidate?.model_path ?? ''} ${modelPath}`.toLowerCase()
  if (VLM_MODEL_MARKERS.some((marker) => haystack.includes(marker))) {
    return 'vision-language'
  }

  return modelPath.trim() ? 'text' : 'unknown'
}

export function getTechniqueOptions(
  modelModality: ModelModality,
  capabilities?: TrainingCapabilitySummary,
): TrainingTechniqueOption[] {
  if (modelModality !== 'vision-language') {
    return TEXT_TECHNIQUE_OPTIONS
  }

  return TEXT_TECHNIQUE_OPTIONS.map((option) => {
    if (option.value !== 'vlm_sft') {
      return {
        ...option,
        enabled: false,
        reason: 'This model looks multimodal, so text-only trainers are hidden until a text model is selected.',
      }
    }

    if (capabilities?.supports_vlm_sft) {
      return {
        ...option,
        enabled: true,
        reason: undefined,
      }
    }

    return {
      ...option,
      enabled: false,
      reason: 'The current gateway does not advertise a VLM SFT training tool yet.',
    }
  })
}

export function getDatasetPlaceholder(technique: TrainingTechnique): string {
  if (technique === 'vlm_sft') {
    return 'Path to multimodal dataset manifest (.jsonl or .json)...'
  }

  return 'Path to .jsonl, .json, .csv, or .parquet file...'
}

export function getDatasetHelpText(technique: TrainingTechnique): string | null {
  if (technique === 'curriculum') {
    return 'Use a scored text dataset or a dataset the backend can score before curriculum staging.'
  }

  if (technique === 'vlm_sft') {
    return 'Use a dataset manifest that pairs image references with instruction and assistant turns.'
  }

  return null
}

export function resolveValidationTechnique(
  technique: TrainingTechnique,
  capabilities?: TrainingCapabilitySummary,
): string | null {
  const supported = capabilities?.supported_validation_techniques ?? []
  const hasExplicitList = supported.length > 0

  if (technique === 'sequential') return null
  if (technique === 'curriculum') {
    return hasExplicitList ? (supported.includes('sft') ? 'sft' : null) : 'sft'
  }
  if (technique === 'vlm_sft') {
    return hasExplicitList ? (supported.includes('vlm_sft') ? 'vlm_sft' : null) : null
  }

  return hasExplicitList ? (supported.includes(technique) ? technique : null) : technique
}

export function supportsSequentialTraining(technique: TrainingTechnique): boolean {
  return technique !== 'curriculum' && technique !== 'vlm_sft' && technique !== 'sequential'
}
