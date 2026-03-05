export interface MCPTool {
  name: string
  description: string
  inputSchema: {
    type: 'object'
    properties: Record<string, JSONSchemaProperty>
    required?: string[]
  }
}

export interface JSONSchemaProperty {
  type: string
  description?: string
  default?: unknown
  enum?: string[]
  items?: JSONSchemaProperty
  properties?: Record<string, JSONSchemaProperty>
}

export interface MCPToolResult {
  success: boolean
  error?: string
  [key: string]: unknown
}

export interface SystemResources {
  gpu: {
    available: boolean
    name?: string
    vram_total_gb?: number
    vram_free_gb?: number
    vram_used_gb?: number
    vram_reserved_gb?: number
    compute_capability?: string
    cuda_version?: string
  }
  ram: {
    total_gb: number
    free_gb: number
    used_gb: number
    percent_used: number
  }
  disk: {
    output_dir: string
    total_gb: number
    free_gb: number
    used_gb: number
  }
}

export interface SetupCheck {
  name: string
  status: 'pass' | 'warn' | 'fail'
  detail: string
}

export interface SetupCheckResult {
  success: boolean
  checks: SetupCheck[]
  all_passed: boolean
}

export interface TrainingJob {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  technique: string
  base_model: string
  dataset_path: string
  output_dir: string
  created_at: string
  started_at?: string
  completed_at?: string
  progress?: TrainingProgress
  error?: string
}

export interface TrainingProgress {
  current_step: number
  max_steps: number
  current_epoch: number
  max_epochs: number
  loss: number
  learning_rate: number
  eval_loss?: number
  grad_norm?: number
  eta_seconds?: number
  percent_complete: number
  gpu_memory_used_gb?: number
  gpu_memory_total_gb?: number
  log_history: Array<{
    loss: number
    learning_rate: number
    epoch: number
    step: number
  }>
}

export interface Deployment {
  deployment_id: string
  model_path: string
  adapter_path?: string
  endpoint: string
  type: 'mcp' | 'api'
  status: 'running' | 'stopped'
}

export interface DatasetInfo {
  dataset_id: string
  file_path: string
  format: string
  row_count: number
  columns: string[]
  technique?: string
  size_bytes: number
}
