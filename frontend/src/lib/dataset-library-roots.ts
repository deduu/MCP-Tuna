import { getDefaultDatasetOutputDir } from '@/lib/dataset-output'

const DATASET_LIBRARY_ROOTS_STORAGE_KEY = 'agentsoul.datasetLibraryRoots'
const LEGACY_DEFAULT_DATASET_LIBRARY_ROOTS = ['data', 'output', 'uploads', 'notebooks']

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

function areSameRoots(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index])
}

function isLegacyDefaultDatasetLibraryRoots(roots: string[]): boolean {
  return areSameRoots(roots, LEGACY_DEFAULT_DATASET_LIBRARY_ROOTS)
}

export function getDefaultDatasetLibraryRoots(): string[] {
  return uniqueRoots([getDefaultDatasetOutputDir()])
}

export function usesDefaultDatasetLibraryRoots(): boolean {
  if (typeof window === 'undefined') {
    return true
  }

  const stored = window.localStorage.getItem(DATASET_LIBRARY_ROOTS_STORAGE_KEY)
  if (!stored) {
    return true
  }

  try {
    const parsed = JSON.parse(stored)
    if (!Array.isArray(parsed)) {
      return true
    }
    const normalized = uniqueRoots(parsed.filter((value): value is string => typeof value === 'string'))
    return (
      areSameRoots(normalized, getDefaultDatasetLibraryRoots()) ||
      isLegacyDefaultDatasetLibraryRoots(normalized)
    )
  } catch {
    return true
  }
}

export function getDatasetLibraryRoots(): string[] {
  if (typeof window === 'undefined') {
    return getDefaultDatasetLibraryRoots()
  }

  const stored = window.localStorage.getItem(DATASET_LIBRARY_ROOTS_STORAGE_KEY)
  if (!stored) {
    return getDefaultDatasetLibraryRoots()
  }

  try {
    const parsed = JSON.parse(stored)
    if (!Array.isArray(parsed)) {
      return getDefaultDatasetLibraryRoots()
    }
    const normalized = uniqueRoots(parsed.filter((value): value is string => typeof value === 'string'))
    if (normalized.length === 0) {
      return getDefaultDatasetLibraryRoots()
    }
    // Migrate the older broad workspace default to the narrower managed dataset root.
    if (isLegacyDefaultDatasetLibraryRoots(normalized)) {
      window.localStorage.removeItem(DATASET_LIBRARY_ROOTS_STORAGE_KEY)
      return getDefaultDatasetLibraryRoots()
    }
    return normalized
  } catch {
    return getDefaultDatasetLibraryRoots()
  }
}

export function setDatasetLibraryRoots(roots: string[]): string[] {
  const normalized = uniqueRoots(roots)
  const nextRoots = normalized.length > 0 ? normalized : getDefaultDatasetLibraryRoots()
  window.localStorage.setItem(DATASET_LIBRARY_ROOTS_STORAGE_KEY, JSON.stringify(nextRoots))
  return nextRoots
}

export function resetDatasetLibraryRoots(): string[] {
  window.localStorage.removeItem(DATASET_LIBRARY_ROOTS_STORAGE_KEY)
  return getDefaultDatasetLibraryRoots()
}

export function parseDatasetLibraryRootsInput(value: string): string[] {
  return uniqueRoots(value.split(/\r?\n|,/))
}

export function formatDatasetLibraryRootsInput(roots: string[]): string {
  return roots.join('\n')
}
