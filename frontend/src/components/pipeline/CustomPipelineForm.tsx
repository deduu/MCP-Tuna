import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { buildDatasetOutputPath } from '@/lib/dataset-output'
import { buildDefaultOutputDir } from '@/lib/training-capabilities'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import { DocumentPathInput } from './DocumentPathInput'
import { ModelPathField, type ModelPathValidation } from './ModelPathField'
import { StepConfigEditor } from './StepConfigEditor'

const PIPELINE_STEPS = [
  'extract',
  'generate',
  'clean',
  'normalize',
  'evaluate',
  'train',
  'deploy',
] as const

const TECHNIQUES = ['sft', 'dpo', 'grpo', 'kto', 'vlm_sft'] as const
const VLM_UNSUPPORTED_STEPS = ['extract', 'generate', 'clean', 'normalize', 'evaluate'] as const

interface CustomPipelineFormProps {
  onSubmit: (args: Record<string, unknown>) => void
  isPending: boolean
}

function resolveTrainUseLora(
  overrides: Record<string, Record<string, unknown>>,
  fallback: boolean,
): boolean {
  const overrideValues = [
    overrides.train?.use_lora,
    overrides['finetune.train']?.use_lora,
  ]

  for (const value of overrideValues) {
    if (typeof value === 'boolean') return value
  }

  return fallback
}

export function CustomPipelineForm({ onSubmit, isPending }: CustomPipelineFormProps) {
  const [selectedSteps, setSelectedSteps] = useState<Set<string>>(new Set())
  const [documentPath, setDocumentPath] = useState('')
  const [validationPath, setValidationPath] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [adapterPath, setAdapterPath] = useState('')
  const [technique, setTechnique] = useState<(typeof TECHNIQUES)[number]>('sft')
  const [useLora, setUseLora] = useState(true)
  const [pushToHub, setPushToHub] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [stepConfig, setStepConfig] = useState<Record<string, Record<string, unknown>>>({})
  const [stepConfigValid, setStepConfigValid] = useState(true)
  const [modelValidation, setModelValidation] = useState<ModelPathValidation | null>(null)
  const [adapterValidation, setAdapterValidation] = useState<ModelPathValidation | null>(null)
  const [quantization, setQuantization] = useState<string>('4bit')

  const hasTrain = selectedSteps.has('train')
  const hasDeploy = selectedSteps.has('deploy')
  const hasGenerate = selectedSteps.has('generate')
  const hasExtract = selectedSteps.has('extract')
  const deployOnly = hasDeploy && !hasTrain
  const isVlmTechnique = technique === 'vlm_sft'
  const usesExistingDataset = !isVlmTechnique && !hasGenerate && !hasExtract && ['clean', 'normalize', 'evaluate', 'train'].some((step) =>
    selectedSteps.has(step),
  )
  const showValidationDatasetPath = hasTrain && technique === 'sft' && !isVlmTechnique
  const trainConsumesExistingDatasetDirectly =
    usesExistingDataset && !selectedSteps.has('clean') && !selectedSteps.has('normalize') && !selectedSteps.has('evaluate')
  const needsDocumentPath = ['extract', 'generate', 'clean', 'normalize', 'evaluate', 'train'].some((step) =>
    selectedSteps.has(step),
  )
  const primaryPathLabel = isVlmTechnique || usesExistingDataset ? 'Dataset Path' : 'Document Path'
  const primaryPathPlaceholder = isVlmTechnique
    ? '/path/to/vlm_dataset.jsonl'
    : usesExistingDataset
      ? '/path/to/train_dataset.jsonl'
      : '/path/to/document-or-dataset'
  const primaryHelperText = isVlmTechnique
    ? 'Use a backend-visible VLM dataset manifest path. Build one in Datasets > Build VLM Dataset Row.'
    : usesExistingDataset
      ? 'Paste an existing dataset path. You can skip extract/generate when the file is already in dataset format.'
      : 'Browse uploads one document to the backend. For an existing dataset, you can also paste its server path.'

  function toggleStep(step: string) {
    setSelectedSteps((prev) => {
      const next = new Set(prev)
      if (next.has(step)) next.delete(step)
      else next.add(step)
      return next
    })
  }

  function handleSubmit() {
    if (needsDocumentPath && !documentPath.trim()) {
      toast.error('Document path is required for the selected steps')
      return
    }
    if (hasDeploy && !hasTrain && !modelPath.trim()) {
      toast.error('Deploy needs a trained model from this pipeline or a model path')
      return
    }
    if (deployOnly && modelValidation?.isAdapter && !adapterPath.trim()) {
      toast.error('That path looks like an adapter folder. Put the base model in Model Path and the adapter folder in Adapter Path.')
      return
    }
    if (isVlmTechnique) {
      const invalidSteps = VLM_UNSUPPORTED_STEPS.filter((step) => selectedSteps.has(step))
      if (invalidSteps.length > 0) {
        toast.error(`VLM pipelines currently support dataset-based train/deploy only. Remove: ${invalidSteps.join(', ')}`)
        return
      }
      if (hasTrain && !documentPath.trim()) {
        toast.error('A VLM dataset path is required for train')
        return
      }
    }

    if (!stepConfigValid) {
      toast.error('Advanced step_config must be valid JSON')
      return
    }
    const overrides = stepConfig
    const effectiveTrainUseLora = resolveTrainUseLora(overrides, useLora)
    const trainOutputDir = buildDefaultOutputDir(technique, false, documentPath.trim() || modelPath.trim())

    const steps: Array<{ tool: string; params: Record<string, unknown> }> = []
    const addStep = (stepId: string, tool: string, params: Record<string, unknown>) => {
      const merged = {
        ...params,
        ...(overrides[stepId] ?? {}),
        ...(overrides[tool] ?? {}),
      }
      steps.push({ tool, params: merged })
    }

    if (selectedSteps.has('extract')) {
      addStep('extract', 'extract.load_document', { file_path: documentPath })
    }

    if (selectedSteps.has('generate')) {
      addStep('generate', 'generate.from_document', { technique, file_path: documentPath })
    } else if (!isVlmTechnique && ['clean', 'normalize', 'evaluate'].some((s) => selectedSteps.has(s))) {
      addStep('load', 'dataset.load', { file_path: documentPath })
    }

    if (selectedSteps.has('clean')) {
      addStep('clean', 'clean.dataset', { data_points: '$prev.data_points' })
    }
    if (selectedSteps.has('normalize')) {
      addStep('normalize', 'normalize.dataset', { data_points: '$prev.data_points', target_format: technique })
    }
    if (selectedSteps.has('evaluate')) {
      addStep('evaluate', 'evaluate.dataset', { data_points: '$prev.data_points' })
    }
    if (selectedSteps.has('train')) {
      if (isVlmTechnique) {
        addStep('train', 'finetune.train_vlm_async', {
          dataset_path: documentPath.trim(),
          output_dir: trainOutputDir,
          ...(modelPath.trim() ? { base_model: modelPath.trim() } : {}),
          use_lora: effectiveTrainUseLora,
        })
      } else if (trainConsumesExistingDatasetDirectly) {
        addStep('train', 'finetune.train', {
          dataset_path: documentPath.trim(),
          output_dir: trainOutputDir,
          ...(modelPath.trim() ? { base_model: modelPath.trim() } : {}),
          ...(showValidationDatasetPath && validationPath.trim() ? { eval_file_path: validationPath.trim() } : {}),
          use_lora: effectiveTrainUseLora,
          ...(pushToHub.trim() ? { push_to_hub: pushToHub.trim() } : {}),
        })
      } else {
        addStep('save', 'dataset.save', {
          data_points: '$prev.data_points',
          output_path: buildDatasetOutputPath(documentPath, 'custom_pipeline'),
          format: 'jsonl',
        })
        addStep('train', 'finetune.train', {
          dataset_path: '$prev.file_path',
          output_dir: trainOutputDir,
          ...(modelPath.trim() ? { base_model: modelPath.trim() } : {}),
          ...(showValidationDatasetPath && validationPath.trim() ? { eval_file_path: validationPath.trim() } : {}),
          use_lora: effectiveTrainUseLora,
          ...(pushToHub.trim() ? { push_to_hub: pushToHub.trim() } : {}),
        })
      }
    }
    if (selectedSteps.has('deploy')) {
      const quantArg = quantization !== 'none' ? { quantization } : {}
      if (selectedSteps.has('train')) {
        addStep(
          'deploy',
          isVlmTechnique ? 'host.deploy_vlm_mcp' : 'host.deploy_mcp',
          effectiveTrainUseLora
            ? {
                model_path: '$prev.base_model',
                adapter_path: '$prev.model_path',
                ...(isVlmTechnique ? {} : quantArg),
              }
            : {
                model_path: '$prev.model_path',
                ...(isVlmTechnique ? {} : quantArg),
              },
        )
      } else if (modelPath.trim()) {
        addStep('deploy', isVlmTechnique ? 'host.deploy_vlm_mcp' : 'host.deploy_mcp', {
          model_path: modelPath.trim(),
          ...(adapterPath.trim() ? { adapter_path: adapterPath.trim() } : {}),
          ...(isVlmTechnique ? {} : quantArg),
        })
      }
    }

    if (steps.length === 0) {
      toast.error('No executable pipeline steps were produced from the current selection')
      return
    }

    const args: Record<string, unknown> = {
      steps: JSON.stringify(steps),
      dry_run: false,
    }
    onSubmit(args)
  }

  return (
    <div className="space-y-4">
      {/* Step selection */}
      <div>
        <label className="text-sm font-medium text-foreground mb-2 block">Steps</label>
        <div className="flex flex-wrap gap-2">
          {PIPELINE_STEPS.map((step) => (
            <label
              key={step}
              className={cn(
                'flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm cursor-pointer transition-colors',
                selectedSteps.has(step)
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-input text-muted-foreground hover:text-foreground',
              )}
            >
              <input
                type="checkbox"
                checked={selectedSteps.has(step)}
                onChange={() => toggleStep(step)}
                className="sr-only"
              />
              {step}
            </label>
          ))}
        </div>
      </div>

      {/* Inputs */}
      <div className={cn('grid gap-3', needsDocumentPath ? 'sm:grid-cols-2' : 'sm:grid-cols-1')}>
        {needsDocumentPath && (
          <div>
            <label className="text-sm font-medium text-foreground mb-1 block">{primaryPathLabel}</label>
            <DocumentPathInput
              value={documentPath}
              onChange={setDocumentPath}
              placeholder={primaryPathPlaceholder}
              disabled={isPending}
              helperText={primaryHelperText}
            />
          </div>
        )}
        <div>
          <label className="text-sm font-medium text-foreground mb-1 block">Model Path</label>
          <ModelPathField
            value={modelPath}
            onChange={setModelPath}
            disabled={isPending}
            onValidationChange={setModelValidation}
            placeholder={deployOnly ? './output/my-model-folder or meta-llama/Llama-3-8B' : 'meta-llama/Llama-3-8B or ~/.cache/huggingface/hub/...'}
            helperText={
              deployOnly
                ? 'Deploy-only accepts either a Hugging Face model ID or a backend-visible model folder. For local deployment, Model Path should point to the model directory, not a single file.'
                : hasTrain && hasDeploy
                  ? useLora
                    ? 'When train and deploy are both selected, this field is the base model for training. Deployment will use the trained adapter output.'
                    : 'When train and deploy are both selected with LoRA disabled, deployment will use the trained model folder directly.'
                  : 'Use a Hugging Face model ID or a backend-visible local model folder.'
            }
          />
        </div>
      </div>

      {showValidationDatasetPath && (
        <div>
          <label className="text-sm font-medium text-foreground mb-1 block">Validation Dataset Path</label>
          <DocumentPathInput
            value={validationPath}
            onChange={setValidationPath}
            placeholder="/path/to/val_dataset.jsonl"
            disabled={isPending}
            helperText="Optional. Use a second JSONL dataset for evaluation during SFT. Leave blank to train without validation."
          />
        </div>
      )}

      {deployOnly && (
        <div>
          <label className="text-sm font-medium text-foreground mb-1 block">Adapter Path</label>
          <ModelPathField
            value={adapterPath}
            onChange={setAdapterPath}
            disabled={isPending}
            validationPurpose="adapter"
            onValidationChange={setAdapterValidation}
            placeholder="./output/custom_pipeline"
            helperText="Optional. Use this when the model folder above is the base model and your fine-tuned weights live in a separate LoRA adapter folder."
          />
        </div>
      )}

      {deployOnly && adapterPath.trim() && adapterValidation?.tone === 'warning' && (
        <p className="text-xs text-amber-400">
          The adapter path does not look like an adapter folder. Deployment may fail unless it is a valid LoRA directory.
        </p>
      )}

      {hasDeploy && (
        <div>
          <label className="text-sm font-medium text-foreground mb-1 block">Quantization</label>
          <select
            value={quantization}
            onChange={(e) => setQuantization(e.target.value)}
            className="flex h-9 w-full max-w-xs rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <option value="4bit">4-bit (recommended, saves memory)</option>
            <option value="8bit">8-bit</option>
            <option value="none">None (full precision)</option>
          </select>
          <p className="mt-1 text-xs text-muted-foreground">
            {isVlmTechnique
              ? 'Text deployments honor quantization here. VLM deploy steps currently use the dedicated multimodal runtime without a quantization override.'
              : '4-bit quantization significantly reduces memory usage. Use "None" only if you have enough VRAM/RAM.'}
          </p>
        </div>
      )}

      {hasTrain && (
        <div className="space-y-3">
          <label className="flex items-center gap-2 text-sm text-foreground">
            <input
              type="checkbox"
              checked={useLora}
              onChange={(e) => setUseLora(e.target.checked)}
              className="h-4 w-4 rounded border-input bg-transparent"
            />
            Train with LoRA adapter
          </label>
          {!isVlmTechnique && (
            <div>
              <label className="text-sm font-medium text-foreground mb-1 block">Push To Hub Repo</label>
              <Input
                value={pushToHub}
                onChange={(e) => setPushToHub(e.target.value)}
                disabled={isPending}
                placeholder="your-org/your-model-name"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Optional. Push the trained model or adapter to this Hugging Face Hub repo after training.
              </p>
            </div>
          )}
        </div>
      )}

      <div>
        <label className="text-sm font-medium text-foreground mb-1 block">Technique</label>
        <select
          value={technique}
          onChange={(e) => setTechnique(e.target.value as (typeof TECHNIQUES)[number])}
          className="flex h-9 w-full max-w-xs rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          {TECHNIQUES.map((t) => (
            <option key={t} value={t}>
              {t === 'vlm_sft' ? 'VLM SFT' : t.toUpperCase()}
            </option>
          ))}
        </select>
        {isVlmTechnique && (
          <p className="mt-1 text-xs text-muted-foreground">
            VLM custom pipelines currently support dataset-based training and deployment. Document extraction and text-only cleanup steps stay disabled for this technique.
          </p>
        )}
      </div>

      {/* Advanced */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground cursor-pointer"
        >
          <ChevronDown
            className={cn('h-3.5 w-3.5 transition-transform', showAdvanced && 'rotate-180')}
          />
          Advanced: step_config
        </button>
        {showAdvanced && (
          <div className="mt-2 space-y-2">
            <StepConfigEditor
              selectedSteps={[...selectedSteps]}
              value={stepConfig}
              onChange={(nextValue, isValid) => {
                setStepConfigValid(isValid)
                setStepConfig(nextValue)
              }}
            />
            <p className="text-xs text-muted-foreground">
              Overrides accept either step ids like <code>generate</code>, <code>train</code>, <code>deploy</code>,
              or full tool names like <code>generate.from_document</code>. Form mode exposes only known common overrides; use JSON mode for less common parameters.
            </p>
          </div>
        )}
      </div>

      <Button
        onClick={handleSubmit}
        disabled={isPending || selectedSteps.size === 0}
      >
        {isPending ? 'Running...' : 'Run Pipeline'}
      </Button>
    </div>
  )
}
