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

// -- HuggingFace Hub search --

export interface HFModel {
  id: string
  author: string | null
  downloads: number
  likes: number
  tags: string[]
  library: string | null
  created_at: string | null
}

export interface HFSearchResult {
  success: boolean
  query: string
  task: string
  models: HFModel[]
  count: number
  error?: string
}

export interface RecommendedModel {
  model_id: string
  size: string
  memory: string
  description: string
}

export interface RecommendResult {
  success: boolean
  use_case: string
  recommendations: RecommendedModel[]
  count: number
  error?: string
}

// -- Auto-prescribe --

export interface AutoPrescribeCandidate {
  rank: number
  model_id: string
  params_b: number
  min_vram_gb: number
  why_recommended: string
  prescribe_config: {
    can_run: boolean
    config: Record<string, unknown>
    dataset_plan: Record<string, unknown>
    resource_snapshot: Record<string, unknown>
    vram_estimate: Record<string, unknown>
    warnings: string[]
    rationale: string[]
  }
}

export interface AutoPrescribeResult {
  success: boolean
  candidates: AutoPrescribeCandidate[]
  filters_applied: {
    available_vram_gb: number
    technique: string
    use_case: string
    models_evaluated: number
    models_fit: number
  }
  error?: string
  technique_warning?: string
  resolved_dataset?: string
}

// -- Confirmation --

export interface ConfirmationRequest {
  tool: string
  arguments: Record<string, unknown>
  message: string
}
