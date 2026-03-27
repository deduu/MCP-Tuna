const DATASET_SUFFIX_ALIASES: Record<string, string> = {
  'clean.dataset': 'clean',
  'clean.deduplicate': 'dedupe',
  'clean.validate_schema': 'schema',
  'clean.remove_short': 'trim',
  'normalize.dataset': 'norm',
  'normalize.merge_fields': 'merge',
  'normalize.standardize_keys': 'keys',
  'normalize.strip_text': 'strip',
  'normalize.remap_fields': 'remap',
  quality_filtered: 'filter',
  custom_pipeline: 'pipe',
  vlm_sft: 'vlm',
  curriculum: 'curr',
}

const TRAINING_TOKEN_ALIASES: Record<string, string> = {
  curriculum: 'curr',
  sequential: 'seq',
  pipeline: 'pipe',
  vlm_sft: 'vlm',
}

function hashToken(value: string): string {
  let hash = 2166136261
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0).toString(36).slice(0, 6)
}

export function sanitizeNameSegment(value: string, fallback = 'item'): string {
  return (
    value
      .trim()
      .replace(/\\/g, '/')
      .split('/')
      .pop()
      ?.replace(/\.[^.]+$/, '')
      .replace(/[^a-zA-Z0-9._-]+/g, '_')
      .replace(/^_+|_+$/g, '') || fallback
  )
}

export function compactNameSegment(value: string, maxLength = 24, fallback = 'item'): string {
  const sanitized = sanitizeNameSegment(value, fallback)
  if (sanitized.length <= maxLength) {
    return sanitized
  }

  const fingerprint = hashToken(sanitized)
  const headLength = Math.max(8, maxLength - fingerprint.length - 1)
  return `${sanitized.slice(0, headLength)}_${fingerprint}`
}

export function compactSourceHint(value: string, maxLength = 24, fallback = 'dataset'): string {
  const withoutUuid = sanitizeNameSegment(value, fallback).replace(/^[0-9a-fA-F-]{36}_/, '')
  return compactNameSegment(withoutUuid, maxLength, fallback)
}

export function compactDatasetSuffix(value: string, maxLength = 14): string {
  const aliased = DATASET_SUFFIX_ALIASES[value] ?? value
  return compactNameSegment(aliased, maxLength, 'dataset')
}

export function compactTrainingPrefix(value: string, maxLength = 18): string {
  const tokens = value
    .split('_')
    .map((token) => TRAINING_TOKEN_ALIASES[token] ?? token)
    .filter(Boolean)
  return compactNameSegment(tokens.join('_'), maxLength, 'train')
}
