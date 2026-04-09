import type { ThinkingMode } from '@/api/types'

export const THINKING_MODE_OPTIONS: Array<{ value: ThinkingMode; label: string }> = [
  { value: 'default', label: 'Default' },
  { value: 'on', label: 'Force On' },
  { value: 'off', label: 'Force Off' },
]

export function formatThinkingModeLabel(mode: ThinkingMode | null | undefined) {
  switch (mode) {
    case 'on':
      return 'Force on'
    case 'off':
      return 'Force off'
    default:
      return 'Default'
  }
}
