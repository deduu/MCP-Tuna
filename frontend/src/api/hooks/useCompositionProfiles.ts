import { useQuery } from '@tanstack/react-query'
import { mcpCall } from '../client'
import type {
  CanonicalSchemaKind,
  CompositionProfile,
  CompositionSchemaAdapter,
} from '../types'

interface CompositionProfilesResponse {
  profiles?: CompositionProfile[]
}

interface CompositionSchemaAdaptersResponse {
  schema_adapters?: CompositionSchemaAdapter[]
}

export function useCompositionProfiles(enabled: boolean = true) {
  return useQuery<CompositionProfile[]>({
    queryKey: ['generate', 'composition', 'profiles'],
    queryFn: async () => {
      const result = await mcpCall<CompositionProfilesResponse>('generate.list_profiles')
      return result.profiles ?? []
    },
    enabled,
    staleTime: 5 * 60_000,
    retry: 1,
  })
}

export function useSchemaAdapters(
  canonicalKind?: CanonicalSchemaKind,
  enabled: boolean = true,
) {
  return useQuery<CompositionSchemaAdapter[]>({
    queryKey: ['generate', 'composition', 'schema-adapters', canonicalKind ?? 'all'],
    queryFn: async () => {
      const result = await mcpCall<CompositionSchemaAdaptersResponse>(
        'generate.list_schema_adapters',
        canonicalKind ? { canonical_kind: canonicalKind } : {},
      )
      return result.schema_adapters ?? []
    },
    enabled,
    staleTime: 5 * 60_000,
    retry: 1,
  })
}
