const DATASET_OUTPUT_DIR_STORAGE_KEY = 'agentsoul.datasetOutputDir'
const DEFAULT_DATASET_OUTPUT_DIR = 'data'

function normalizeSegment(value: string): string {
  return value
    .trim()
    .replace(/\\/g, '/')
    .replace(/\/+/g, '/')
    .replace(/\/$/, '')
}

function sanitizeStem(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'dataset'
}

export function getDefaultDatasetOutputDir(): string {
  if (typeof window === 'undefined') {
    return DEFAULT_DATASET_OUTPUT_DIR
  }

  const stored = window.localStorage.getItem(DATASET_OUTPUT_DIR_STORAGE_KEY)
  return normalizeSegment(stored || '') || DEFAULT_DATASET_OUTPUT_DIR
}

export function setDefaultDatasetOutputDir(dir: string): string {
  const normalized = normalizeSegment(dir) || DEFAULT_DATASET_OUTPUT_DIR
  window.localStorage.setItem(DATASET_OUTPUT_DIR_STORAGE_KEY, normalized)
  return normalized
}

export function resetDefaultDatasetOutputDir(): string {
  window.localStorage.removeItem(DATASET_OUTPUT_DIR_STORAGE_KEY)
  return DEFAULT_DATASET_OUTPUT_DIR
}

export function buildDatasetOutputPath(
  sourcePath: string,
  suffix: string,
  extension = 'jsonl',
  outputDir = getDefaultDatasetOutputDir(),
): string {
  const normalizedSource = sourcePath.replace(/\\/g, '/')
  const filename = normalizedSource.split('/').pop() || sourcePath
  const dot = filename.lastIndexOf('.')
  const stem = dot >= 0 ? filename.slice(0, dot) : filename
  const readableStem = stem.replace(/^[0-9a-fA-F-]{36}_/, '')
  const safeStem = sanitizeStem(readableStem)
  const safeSuffix = sanitizeStem(suffix)
  const dir = normalizeSegment(outputDir) || DEFAULT_DATASET_OUTPUT_DIR
  const safeExtension = extension.replace(/^\.+/, '') || 'jsonl'
  return `${dir}/${safeStem}_${safeSuffix}.${safeExtension}`
}
