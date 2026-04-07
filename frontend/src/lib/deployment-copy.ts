export const DEPLOYMENT_ADAPTER_PATH_LABEL = 'Final Adapter Path'

export function describeDeploymentTarget(): string {
  return 'Deploy either a model folder directly, or a matching base model plus one final adapter folder.'
}

export function describeDeploymentNameDefault(): string {
  return 'Optional. Defaults to the model name, or a base-model plus final-adapter label for adapter-backed deployments.'
}

export function describeDeploymentAdapterField(): string {
  return 'Optional. Use this when Model Path is the matching base model and the deployed fine-tuned output lives in a separate final adapter folder.'
}

export function describeAdapterBackedOutput(): string {
  return 'Adapter-backed output detected. Deploy with the matching base model plus this final adapter folder.'
}

export function describeDirectModelOutput(): string {
  return 'Merged or full-model output detected. Deploy this folder directly as Model Path.'
}

export function describeAdapterBackedInference(): string {
  return 'Adapter-backed run detected. Inference uses the matching base model plus final adapter automatically.'
}

export function describeModelPathLooksLikeAdapter(): string {
  return 'That path looks like an adapter folder. Put the matching base model in Model Path and the final adapter folder in Final Adapter Path.'
}

export function describeInvalidDeploymentAdapterPath(): string {
  return 'The final adapter path does not look like an adapter folder. Deployment may fail unless it is a valid adapter directory.'
}

export function describeTrainAndDeployModelPath(useAdapter: boolean): string {
  if (useAdapter) {
    return 'When train and deploy are both selected, this field is the base model for training. Deployment will use the final adapter output from training.'
  }

  return 'When train and deploy are both selected without adapter-based training, deployment will use the trained model folder directly.'
}

export function describeDeployAfterTraining(useAdapter: boolean): string {
  if (useAdapter) {
    return 'Deployment will load the selected base model plus the final adapter output from training.'
  }

  return 'Deployment will load the trained model folder directly because adapter-based training is disabled.'
}
