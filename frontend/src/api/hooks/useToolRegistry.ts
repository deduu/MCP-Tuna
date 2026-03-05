import { useQuery } from '@tanstack/react-query'
import { mcpListTools } from '../client'
import type { MCPTool } from '../types'

export function useToolRegistry() {
  return useQuery<MCPTool[]>({
    queryKey: ['tools', 'registry'],
    queryFn: mcpListTools,
    staleTime: 5 * 60 * 1000,
    retry: 2,
  })
}
