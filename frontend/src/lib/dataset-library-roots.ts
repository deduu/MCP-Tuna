const DATASET_LIBRARY_ROOTS_STORAGE_KEY = 'agentsoul.datasetLibraryRoots'
export const DEFAULT_DATASET_LIBRARY_ROOTS = ['data', 'output', 'uploads', 'notebooks']

function normalizeRoot(value: string): string {
  const normalized = value
    .trim()
    .replace(/\\/g, '/')
    .replace(/\/+/g, '/')

  if (!normalized) {
    return ''
  }

  if (/^[A-Za-z]:\/$/.test(normalized) || normalized === '/') {
    return normalized
  }

  return normalized.replace(/\/$/, '')
}

function uniqueRoots(values: string[]): string[] {
  return Array.from(new Set(values.map(normalizeRoot).filter(Boolean)))
}

export function getDatasetLibraryRoots(): string[] {
  if (typeof window === 'undefined') {
    return [...DEFAULT_DATASET_LIBRARY_ROOTS]
  }

  const stored = window.localStorage.getItem(DATASET_LIBRARY_ROOTS_STORAGE_KEY)
  if (!stored) {
    return [...DEFAULT_DATASET_LIBRARY_ROOTS]
  }

  try {
    const parsed = JSON.parse(stored)
    if (!Array.isArray(parsed)) {
      return [...DEFAULT_DATASET_LIBRARY_ROOTS]
    }
    const normalized = uniqueRoots(parsed.filter((value): value is string => typeof value === 'string'))
    return normalized.length > 0 ? normalized : [...DEFAULT_DATASET_LIBRARY_ROOTS]
  } catch {
    return [...DEFAULT_DATASET_LIBRARY_ROOTS]
  }
}

export function setDatasetLibraryRoots(roots: string[]): string[] {
  const normalized = uniqueRoots(roots)
  const nextRoots = normalized.length > 0 ? normalized : [...DEFAULT_DATASET_LIBRARY_ROOTS]
  window.localStorage.setItem(DATASET_LIBRARY_ROOTS_STORAGE_KEY, JSON.stringify(nextRoots))
  return nextRoots
}

export function resetDatasetLibraryRoots(): string[] {
  window.localStorage.removeItem(DATASET_LIBRARY_ROOTS_STORAGE_KEY)
  return [...DEFAULT_DATASET_LIBRARY_ROOTS]
}

export function parseDatasetLibraryRootsInput(value: string): string[] {
  return uniqueRoots(value.split(/\r?\n|,/))
}

export function formatDatasetLibraryRootsInput(roots: string[]): string {
  return roots.join('\n')
}
