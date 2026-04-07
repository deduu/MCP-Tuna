export const INITIAL_ADAPTER_LABEL = 'Initial Adapter Path'
export const TRAIN_WITH_ADAPTERS_LABEL = 'Train with adapters'
export const FINE_TUNE_NAMESPACE_DESCRIPTION = 'Adapter-based training (SFT/DPO/GRPO/KTO)'
export const FINE_TUNE_QUICK_ACTION_DESCRIPTION = 'Fine-tune with adapters and preference training'

export function describeInitialAdapterRecipeHint(fieldLabel: string = INITIAL_ADAPTER_LABEL): string {
  return `This recipe assumes you continue from your best SFT-stage adapter. Set ${fieldLabel} before running.`
}

export function describeInitialAdapterRequirement(): string {
  return 'Initial adapter path requires adapter-based training to stay enabled'
}

export function describeContinueFromAdapter(): string {
  return 'Optional. Continue training from an existing adapter instead of starting from the base model only.'
}

export function describeContinueFromAdapterForPipeline(): string {
  return 'Optional. Continue training from an existing adapter before running the selected preference/SFT stage.'
}

export function describeCurriculumAdvancedControls(): string {
  return 'Curriculum async training currently exposes stage and scoring controls, plus adapter and quantization settings.'
}
