import { useMutation } from '@tanstack/react-query'
import { mcpCall } from '../client'
import type { MCPToolResult } from '../types'

interface ToolExecutionParams {
  toolName: string
  args: Record<string, unknown>
}

export function useToolExecution() {
  return useMutation<MCPToolResult, Error, ToolExecutionParams>({
    mutationFn: ({ toolName, args }) => mcpCall<MCPToolResult>(toolName, args),
  })
}
