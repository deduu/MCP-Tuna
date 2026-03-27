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
  enum?: Array<string | number | boolean>
  format?: string
  items?: JSONSchemaProperty
  properties?: Record<string, JSONSchemaProperty>
}

export interface MCPToolResult {
  success?: boolean
  error?: string
  [key: string]: unknown
}

export type ModelModality = 'text' | 'vision-language' | 'unknown'

export type TrainingTechnique =
  | 'sft'
  | 'dpo'
  | 'grpo'
  | 'kto'
  | 'curriculum'
  | 'vlm_sft'
  | 'sequential'

export interface GPUInfo {
  index?: number
  available: boolean
  name?: string
  vram_total_gb?: number
  vram_free_gb?: number
  vram_used_gb?: number
  vram_reserved_gb?: number
  compute_capability?: string
  cuda_version?: string
}

export interface SystemResources {
  gpu: GPUInfo
  gpus?: GPUInfo[]
  gpu_count?: number
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
  category?: 'provider' | 'system' | 'package'
  required?: boolean
  action_path?: string
  action_label?: string
}

export interface SetupCheckResult {
  success: boolean
  checks: SetupCheck[]
  all_passed: boolean
}

export interface SystemHealthResult {
  success: boolean
  status: 'green' | 'yellow' | 'red'
  resources: SystemResources
  active_training_jobs: number
  active_deployments: number
  warnings: string[]
}

export interface GatewayConfigResult {
  success: boolean
  config: {
    finetuning?: {
      base_model?: string
      [key: string]: unknown
    }
    [key: string]: unknown
  }
  env: {
    OPENAI_API_KEY?: string | null
    OPENAI_API_BASE?: string | null
    ANTHROPIC_API_KEY?: string | null
    ANTHROPIC_API_BASE?: string | null
    GOOGLE_API_KEY?: string | null
    HF_TOKEN?: string | null
    [key: string]: string | null | undefined
  }
}

export interface TrainingJob {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  technique?: string
  trainer_type?: string
  base_model: string
  dataset_path?: string
  output_dir: string
  created_at: string
  started_at?: string
  completed_at?: string
  elapsed_seconds?: number
  progress?: TrainingProgress
  error?: string
  result?: Record<string, unknown>
  config_summary?: Record<string, unknown>
}

export interface TrainingProgress {
  current_step: number
  max_steps: number
  current_epoch: number
  max_epochs: number
  loss?: number | null
  learning_rate?: number | null
  eval_loss?: number | null
  grad_norm?: number | null
  eta_seconds?: number | null
  percent_complete: number
  gpu_memory_used_gb?: number
  gpu_memory_total_gb?: number
  current_stage?: string
  status_message?: string
  stage_current?: number
  stage_total?: number
  stage_unit?: string
  log_history: Array<{
    loss: number
    learning_rate: number
    epoch: number
    step: number
  }>
}

export interface Deployment {
  deployment_id: string
  name?: string
  system_prompt?: string | null
  model_path: string
  adapter_path?: string
  endpoint: string
  type: 'mcp' | 'api'
  status: 'running' | 'stopped'
  transport?: string
  modality?: ModelModality
  routes?: string[]
  created_at?: string
  updated_at?: string
  stopped_at?: string
}

export interface DatasetInfo {
  dataset_id: string
  file_path: string
  format: string
  row_count: number
  columns: string[]
  technique?: string
  size_bytes: number
  modified_at?: string
  object_key?: string
  object_url?: string
}

export interface ConversationMessage {
  sequence: number
  role: 'user' | 'assistant'
  content: string | Array<Record<string, unknown>>
}

export interface DeploymentConversationSummary {
  conversation_id: string
  title?: string | null
  deployment_id?: string | null
  modality: ModelModality
  endpoint?: string | null
  model_path?: string | null
  adapter_path?: string | null
  message_count: number
  created_at?: string
  updated_at?: string
}

export interface DeploymentConversation extends DeploymentConversationSummary {
  system_prompt?: string | null
  messages: ConversationMessage[]
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

export interface LocalModelCandidate {
  id: string
  model_path: string
  usable_for?: string[]
  modality?: ModelModality
  supported_techniques?: TrainingTechnique[]
}

export interface TrainingCapabilitySummary {
  available_techniques: TrainingTechnique[]
  supports_vlm_sft: boolean
  supported_validation_techniques: string[]
}

export interface DeploymentBrowseRoot {
  id: string
  label: string
  path: string
  exists: boolean
}

export interface DeploymentBrowseEntry {
  name: string
  path: string
  absolute_path: string
  type: 'directory' | 'file'
  selectable: boolean
}

export interface DeploymentBrowseResult {
  success: boolean
  root_id: string
  root_path: string
  current_path: string
  current_absolute_path: string
  parent_path?: string | null
  entries: DeploymentBrowseEntry[]
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
