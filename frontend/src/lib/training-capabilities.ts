import type {
  LocalModelCandidate,
  ModelModality,
  TrainingCapabilitySummary,
  TrainingTechnique,
} from '@/api/types'

export type DifficultyOrder = 'easy_first' | 'hard_first'
export type QuantizationOption = '4bit' | 'none'

export interface TrainingTechniqueOption {
  value: TrainingTechnique
  label: string
  description: string
  enabled: boolean
  reason?: string
}

function sanitizeOutputSegment(value: string): string {
  return value
    .trim()
    .replace(/\\/g, '/')
    .split('/')
    .pop()
    ?.replace(/\.[^.]+$/, '')
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    || ''
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

export function buildDefaultOutputDir(
  technique: TrainingTechnique,
  sequential: boolean,
  sourcePath?: string,
): string {
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
  const prefix = sequential ? `sequential_${technique}` : technique
  const sourceSuffix = sanitizeOutputSegment(sourcePath || '')

  return `./output/${prefix}${sourceSuffix ? `_${sourceSuffix}` : ''}_${stamp}`
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
