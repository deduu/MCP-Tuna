export interface NamespaceInfo {
  id: string
  label: string
  description: string
  color: string
  iconName: string
}

export const NAMESPACES: NamespaceInfo[] = [
  { id: 'system', label: 'System', description: 'Resource checking & environment setup', color: 'var(--color-ns-system)', iconName: 'Server' },
  { id: 'extract', label: 'Extract', description: 'Document loading & parsing', color: 'var(--color-ns-extract)', iconName: 'FileText' },
  { id: 'generate', label: 'Generate', description: 'SFT/DPO/GRPO/KTO data generation', color: 'var(--color-ns-generate)', iconName: 'Sparkles' },
  { id: 'clean', label: 'Clean', description: 'Deduplication & schema validation', color: 'var(--color-ns-clean)', iconName: 'Brush' },
  { id: 'normalize', label: 'Normalize', description: 'Format conversion & key standardization', color: 'var(--color-ns-normalize)', iconName: 'ArrowRightLeft' },
  { id: 'evaluate', label: 'Evaluate', description: 'Quality scoring & filtering', color: 'var(--color-ns-evaluate)', iconName: 'BarChart3' },
  { id: 'dataset', label: 'Dataset', description: 'Load, save, preview, split & merge', color: 'var(--color-ns-dataset)', iconName: 'Database' },
  { id: 'finetune', label: 'Fine-tune', description: 'LoRA training (SFT/DPO/GRPO/KTO)', color: 'var(--color-ns-finetune)', iconName: 'FlaskConical' },
  { id: 'test', label: 'Test', description: 'Model inference & comparison', color: 'var(--color-ns-test)', iconName: 'FlaskConical' },
  { id: 'validate', label: 'Validate', description: 'Model discovery & info', color: 'var(--color-ns-validate)', iconName: 'ShieldCheck' },
  { id: 'host', label: 'Host', description: 'Model deployment as MCP or API', color: 'var(--color-ns-host)', iconName: 'Globe' },
  { id: 'workflow', label: 'Workflow', description: 'End-to-end pipeline execution', color: 'var(--color-ns-workflow)', iconName: 'Workflow' },
  { id: 'orchestration', label: 'Orchestration', description: 'Agent trajectory collection & scoring', color: 'var(--color-ns-orchestration)', iconName: 'Bot' },
  { id: 'judge', label: 'Judge', description: 'LLM-as-a-judge evaluation', color: 'var(--color-ns-judge)', iconName: 'Scale' },
  { id: 'ft_eval', label: 'FT Eval', description: 'Fine-tuned model evaluation', color: 'var(--color-ns-ft-eval)', iconName: 'Star' },
  { id: 'evaluate_model', label: 'Model Eval', description: 'Model benchmarking & comparison', color: 'var(--color-ns-evaluate-model)', iconName: 'MonitorCheck' },
]

export const NAMESPACE_MAP = Object.fromEntries(NAMESPACES.map(ns => [ns.id, ns]))

export function getNamespaceFromToolName(toolName: string): string {
  const dotIndex = toolName.indexOf('.')
  return dotIndex >= 0 ? toolName.substring(0, dotIndex) : toolName
}

export function getToolShortName(toolName: string): string {
  const dotIndex = toolName.indexOf('.')
  return dotIndex >= 0 ? toolName.substring(dotIndex + 1) : toolName
}
