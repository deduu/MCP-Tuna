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
export type ThinkingMode = 'default' | 'on' | 'off'

export type TrainingTechnique =
  | 'sft'
  | 'dpo'
  | 'grpo'
  | 'kto'
  | 'curriculum'
  | 'vlm_sft'
  | 'sequential'

export type PreferenceTechnique = 'dpo' | 'grpo' | 'kto'

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
  thinking_mode?: ThinkingMode
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

export type CompositionMode = 'general' | 'coding' | 'agent'
export type CanonicalSchemaKind = 'text_sft' | 'preference_pair' | 'reward_group' | 'binary_label'

export interface CompositionCapabilityTarget {
  capability: string
  weight_percent: number
  min_rows: number
  enabled: boolean
}

export interface CompositionProfile {
  name: string
  mode: CompositionMode
  description: string
  default_objective: string
  allowed_objectives: string[]
  capability_targets: CompositionCapabilityTarget[]
}

export interface CompositionSchemaAdapter {
  name: string
  canonical_kind: CanonicalSchemaKind
  description: string
  field_map: Record<string, string>
  defaults: Record<string, unknown>
  strict: boolean
  is_default?: boolean
  expected_canonical_kind?: CanonicalSchemaKind
}

export interface CompositionSourceSummary {
  input_path: string
  resolved_path: string
  kind: string
  exists: boolean
  estimated_chunks: number
  estimated_text_chars: number
  scanned_files: number
  code_files: number
  truncated: boolean
  warnings: string[]
}

export interface CompositionSourceTotals {
  valid_sources: number
  estimated_chunks: number
  estimated_text_chars: number
  scanned_files: number
  code_files: number
}

export interface CompositionPreviewResult {
  success: boolean
  profile_name: string
  mode: CompositionMode
  objective: string
  allowed_objectives: string[]
  row_target: number
  requested_mix: Record<string, number>
  resolved_mix: Record<string, number>
  row_plan: Record<string, number>
  schema_adapter: CompositionSchemaAdapter
  source_summaries: CompositionSourceSummary[]
  source_totals: CompositionSourceTotals
  warnings: string[]
  error?: string
}

export interface CompositionDatasetSaveResult {
  success?: boolean
  file_path: string
  format: string
  row_count: number
}

export interface CompositionComposeResult {
  success: boolean
  profile_name: string
  mode: CompositionMode
  objective: string
  row_target: number
  row_count: number
  requested_mix: Record<string, number>
  resolved_mix: Record<string, number>
  achieved_mix: Record<string, number>
  row_plan: Record<string, number>
  dataset: CompositionDatasetSaveResult
  manifest_path: string
  warnings: string[]
  generation_errors?: Array<Record<string, unknown>>
  error?: string
}

export interface CompositionValidationResult {
  success: boolean
  status: 'pass' | 'warn' | 'fail'
  dataset_path: string
  manifest_path: string
  profile_name: string
  mode: CompositionMode
  objective: string
  row_count: number
  expected_columns: string[]
  columns: string[]
  requested_mix: Record<string, number>
  resolved_mix: Record<string, number>
  manifest_achieved_mix: Record<string, number>
  dataset_achieved_mix: Record<string, number>
  capability_counts: Record<string, number>
  source_ref_coverage: number
  warnings: string[]
  errors: string[]
  error?: string
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
  thinking_mode?: ThinkingMode
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
  supports_preference_dataset_analysis: boolean
  supported_validation_techniques: string[]
}

export interface PreferenceDatasetTextStats {
  nonempty_count: number
  empty_count: number
  unique_count: number
  unique_ratio: number
  duplicate_rows: number
  length: {
    avg: number
    min: number
    max: number
    p95: number
  }
  top_repeated: Array<{
    preview: string
    count: number
  }>
}

export interface PreferenceDatasetDpoStats {
  chosen_stats: PreferenceDatasetTextStats
  rejected_stats: PreferenceDatasetTextStats
  duplicate_pair_rows: number
  duplicate_pair_ratio: number
  identical_pair_rows: number
  identical_pair_ratio: number
  avg_token_overlap: number
  high_overlap_rows: number
  high_overlap_ratio: number
  hard_negative_rows: number
  hard_negative_ratio: number
  low_overlap_rows: number
  low_overlap_ratio: number
  dominant_rejected_count: number
  dominant_rejected_ratio: number
  dominant_rejected_preview: string
  avg_rows_per_prompt: number
  avg_rejected_variants_per_prompt: number
  multi_variant_prompt_count: number
  multi_variant_prompt_ratio: number
  short_chosen_rows: number
  short_rejected_rows: number
}

export interface PreferenceDatasetGrpoStats {
  response_stats: PreferenceDatasetTextStats
  avg_responses_per_row: number
  avg_unique_responses_per_row: number
  single_response_rows: number
  invalid_rows: number
  mismatched_rows: number
  identical_response_rows: number
  identical_response_ratio: number
  zero_reward_variance_rows: number
  zero_reward_variance_ratio: number
  reward_stats: {
    count: number
    min: number
    max: number
    mean: number
    stdev: number
  }
  top_repeated_responses: Array<{
    preview: string
    count: number
  }>
}

export interface PreferenceDatasetKtoStats {
  completion_stats: PreferenceDatasetTextStats
  positive_count: number
  negative_count: number
  positive_ratio: number
  negative_ratio: number
  invalid_label_rows: number
}

export interface PreferenceTrainingGuidance {
  headline: string
  starting_recipe: Record<string, string | number | boolean>
  hidden_factors: string[]
  recommended_actions: string[]
  missed_contributors: string[]
  prompt_diversity_ratio: number
}

export interface PreferenceDatasetAnalysisResult {
  success: boolean
  dataset_path: string
  technique_detected?: string | null
  technique_analyzed: PreferenceTechnique
  columns: string[]
  row_count: number
  analyzed_row_count: number
  truncated: boolean
  status: 'pass' | 'warn'
  risk_level: 'low' | 'medium' | 'high'
  prompt_stats: PreferenceDatasetTextStats
  warnings: string[]
  recommendations: string[]
  guidance?: PreferenceTrainingGuidance
  dpo?: PreferenceDatasetDpoStats
  grpo?: PreferenceDatasetGrpoStats
  kto?: PreferenceDatasetKtoStats
  error?: string
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
  modified_at?: string
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
