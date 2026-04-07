import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, Sparkles } from 'lucide-react'
import { toast } from 'sonner'
import type { AutoPrescribeCandidate, TrainingTechnique } from '@/api/types'
import {
  useAutoSuggestModel,
  useLocalModelCandidates,
  useStartTraining,
  useTrainingCapabilities,
} from '@/api/hooks/useTraining'
import { useRunPipeline } from '@/api/hooks/usePipeline'
import { useDatasets } from '@/api/hooks/useDatasets'
import { mcpCall } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { PreferenceDatasetAnalysisCard } from '@/components/shared/PreferenceDatasetAnalysisCard'
import {
  buildDefaultOutputDir,
  DEFAULT_GRADIENT_ACCUMULATION_STEPS,
  DEFAULT_LEARNING_RATE,
  DEFAULT_LORA_ALPHA,
  DEFAULT_LORA_DROPOUT,
  DEFAULT_LORA_R,
  DEFAULT_NUM_EPOCHS,
  DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
  defaultsToLoraTraining,
  extractPreferenceStartingRecipePatch,
  getDatasetHelpText,
  getDatasetPlaceholder,
  getTechniqueOptions,
  inferModelModality,
  isPreferenceTechnique,
  resolveValidationTechnique,
  supportsAdapterInitialization,
  supportsSequentialTraining,
} from '@/lib/training-capabilities'
import {
  describeContinueFromAdapter,
  describeCurriculumAdvancedControls,
  describeInitialAdapterRecipeHint,
  describeInitialAdapterRequirement,
  INITIAL_ADAPTER_LABEL,
} from '@/lib/training-copy'
import { cn } from '@/lib/utils'
import { ModelBrowser } from './ModelBrowser'
import { TrainingDatasetField } from './TrainingDatasetField'
import { TrainingTechniqueSelector } from './TrainingTechniqueSelector'
import { ModelPathField } from '@/components/pipeline/ModelPathField'

interface NewTrainingPanelProps {
  open: boolean
  onToggle: () => void
  onSubmit: () => void
  modelPath: string
  onModelPathChange: (value: string) => void
}

const TRAINING_RECIPE_OPTIONS = [
  { value: 'none', label: 'No Recipe', helper: 'Use the generic MCP Tuna training path.' },
  {
    value: 'tiny_reasoning_stage_1',
    label: 'Tiny Reasoning Stage 1',
    helper: 'Adds the stage-1 system prompt preset for non-reasoning SFT.',
  },
  {
    value: 'tiny_reasoning_stage_2',
    label: 'Tiny Reasoning Stage 2',
    helper: 'Adds reasoning system-prompt formatting and registers <think> tags.',
  },
  {
    value: 'tiny_reasoning_stage_3',
    label: 'Tiny Reasoning Stage 3',
    helper: 'Applies the stage-3 DPO preset defaults when supported by the backend.',
  },
] as const

export function NewTrainingPanel({
  open,
  onToggle: _onToggle,
  onSubmit,
  modelPath,
  onModelPathChange,
}: NewTrainingPanelProps) {
  const [technique, setTechnique] = useState<TrainingTechnique>('sft')
  const [sequential, setSequential] = useState(false)
  const [datasetPath, setDatasetPath] = useState('')
  const [evalDatasetPath, setEvalDatasetPath] = useState('')
  const [initAdapterPath, setInitAdapterPath] = useState('')
  const [benchmarkAfterTraining, setBenchmarkAfterTraining] = useState(false)
  const [optimizeAcrossRuns, setOptimizeAcrossRuns] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [outputDir, setOutputDir] = useState(() => buildDefaultOutputDir('sft', false))
  const [outputDirCustomized, setOutputDirCustomized] = useState(false)
  const [quantization, setQuantization] = useState<'4bit' | 'none'>('4bit')

  const [learningRate, setLearningRate] = useState(String(DEFAULT_LEARNING_RATE))
  const [epochs, setEpochs] = useState(String(DEFAULT_NUM_EPOCHS))
  const [batchSize, setBatchSize] = useState(String(DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE))
  const [loraR, setLoraR] = useState(String(DEFAULT_LORA_R))
  const [loraAlpha, setLoraAlpha] = useState(String(DEFAULT_LORA_ALPHA))
  const [loraDropout, setLoraDropout] = useState(String(DEFAULT_LORA_DROPOUT))

  const [warmupRatio, setWarmupRatio] = useState('0')
  const [weightDecay, setWeightDecay] = useState('0.01')
  const [gradAccum, setGradAccum] = useState(String(DEFAULT_GRADIENT_ACCUMULATION_STEPS))
  const [maxSeqLength, setMaxSeqLength] = useState('2048')
  const [numStages, setNumStages] = useState('3')
  const [scoreColumn, setScoreColumn] = useState('weighted_score')
  const [difficultyOrder, setDifficultyOrder] = useState<'easy_first' | 'hard_first'>('easy_first')
  const [trainingRecipe, setTrainingRecipe] = useState<(typeof TRAINING_RECIPE_OPTIONS)[number]['value']>('none')

  const [schemaValid, setSchemaValid] = useState<'pass' | 'warn' | null>(null)
  const [qualityValid, setQualityValid] = useState<'pass' | 'warn' | null>(null)
  const [suggestions, setSuggestions] = useState<AutoPrescribeCandidate[]>([])

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const { data: datasets = [] } = useDatasets()
  const { data: localCandidates = [] } = useLocalModelCandidates()
  const { data: trainingCapabilities } = useTrainingCapabilities()
  const startTraining = useStartTraining()
  const runPipeline = useRunPipeline()
  const autoSuggest = useAutoSuggestModel()

  const selectedCandidate = useMemo(
    () => localCandidates.find((candidate) => candidate.model_path === modelPath || candidate.id === modelPath),
    [localCandidates, modelPath],
  )
  const modelModality = useMemo(
    () => inferModelModality(modelPath, selectedCandidate),
    [modelPath, selectedCandidate],
  )
  const techniqueOptions = useMemo(
    () => getTechniqueOptions(modelModality, trainingCapabilities),
    [modelModality, trainingCapabilities],
  )
  const selectedTechniqueOption = techniqueOptions.find((option) => option.value === technique) ?? techniqueOptions[0]
  const sequentialAllowed = supportsSequentialTraining(technique)
  const supportsInitAdapter = !sequential && supportsAdapterInitialization(technique)
  const validationTechnique = resolveValidationTechnique(technique, trainingCapabilities)
  const datasetPlaceholder = getDatasetPlaceholder(technique)
  const datasetHelpText = getDatasetHelpText(technique)
  const vlmSupportMissing = modelModality === 'vision-language' && !trainingCapabilities?.supports_vlm_sft
  const autoSuggestDisabled = !datasetPath || autoSuggest.isPending || technique === 'vlm_sft'
  const showEvalDatasetField = technique === 'sft' && !sequential
  const supportsBenchmarkWorkflow = showEvalDatasetField
  const preferenceTechnique = isPreferenceTechnique(technique) ? technique : null
  const showPreferenceDatasetAnalysis =
    preferenceTechnique !== null &&
    Boolean(datasetPath.trim()) &&
    Boolean(trainingCapabilities?.supports_preference_dataset_analysis)
  const isSubmitting = startTraining.isPending || runPipeline.isPending
  const canSubmit = Boolean(modelPath && datasetPath && selectedTechniqueOption?.enabled && !isSubmitting)
  const submitLabel = isSubmitting
    ? 'Starting...'
    : benchmarkAfterTraining
      ? optimizeAcrossRuns
        ? 'Start Optimization'
        : 'Start Training + Benchmark'
      : 'Start Training'
  const schemaTechniqueLabel = technique === 'curriculum'
    ? 'CURRICULUM (SFT schema)'
    : technique === 'vlm_sft'
      ? 'VLM SFT'
      : technique.toUpperCase()

  useEffect(() => {
    if (techniqueOptions.length === 0) return

    const hasCurrentOption = techniqueOptions.some((option) => option.value === technique)
    if (!hasCurrentOption) {
      setTechnique(techniqueOptions[0].value)
    }
  }, [technique, techniqueOptions])

  useEffect(() => {
    if (!outputDirCustomized) {
      setOutputDir(buildDefaultOutputDir(technique, sequential, datasetPath))
    }
  }, [technique, sequential, datasetPath, outputDirCustomized])

  useEffect(() => {
    if (!sequentialAllowed && sequential) {
      setSequential(false)
    }
  }, [sequential, sequentialAllowed])

  useEffect(() => {
    if (!supportsInitAdapter && initAdapterPath) {
      setInitAdapterPath('')
    }
  }, [initAdapterPath, supportsInitAdapter])

  useEffect(() => {
    if (!supportsBenchmarkWorkflow && (benchmarkAfterTraining || optimizeAcrossRuns)) {
      setBenchmarkAfterTraining(false)
      setOptimizeAcrossRuns(false)
    }
  }, [benchmarkAfterTraining, optimizeAcrossRuns, supportsBenchmarkWorkflow])

  useEffect(() => {
    if (!datasetPath || !validationTechnique) {
      setSchemaValid(null)
      setQualityValid(null)
      return
    }

    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      mcpCall<{ success: boolean; technique_detected?: string; missing_columns?: string[] }>(
        'validate.schema',
        { dataset_path: datasetPath, technique: validationTechnique },
      )
        .then((result) => {
          setSchemaValid(result.success ? 'pass' : 'warn')
          if (!result.success && (result.missing_columns ?? []).length > 0) {
            const detected = result.technique_detected
              ? `${result.technique_detected.toUpperCase()} format`
              : 'unknown format'
            toast.warning(
              `Dataset is ${detected} - missing ${(result.missing_columns ?? []).join(', ')} for ${schemaTechniqueLabel}`,
            )
          }
        })
        .catch(() => setSchemaValid('warn'))

      if (technique !== 'vlm_sft') {
        mcpCall<{ success: boolean }>('validate.data_quality', { dataset_path: datasetPath })
          .then((result) => setQualityValid(result.success ? 'pass' : 'warn'))
          .catch(() => setQualityValid('warn'))
      } else {
        setQualityValid(null)
      }
    }, 800)

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [datasetPath, validationTechnique, schemaTechniqueLabel, technique])

  function applySuggestedConfig(candidate: AutoPrescribeCandidate) {
    onModelPathChange(candidate.model_id)
    const config = candidate.prescribe_config?.config ?? {}

    if (config.learning_rate) setLearningRate(String(config.learning_rate))
    if (config.num_epochs) setEpochs(String(config.num_epochs))
    if (config.per_device_train_batch_size) setBatchSize(String(config.per_device_train_batch_size))
    if (config.lora_r) setLoraR(String(config.lora_r))
    if (config.lora_alpha) setLoraAlpha(String(config.lora_alpha))
    if (config.gradient_accumulation_steps) setGradAccum(String(config.gradient_accumulation_steps))
    if (config.max_seq_length) setMaxSeqLength(String(config.max_seq_length))

    setSuggestions([])
    toast.success(`Applied ${candidate.model_id} with optimized config`)
  }

  function applyPreferenceStartingRecipe(recipe: Record<string, string | number | boolean>) {
    const patch = extractPreferenceStartingRecipePatch(recipe)

    if (patch.learning_rate !== undefined) {
      setLearningRate(String(patch.learning_rate))
    }
    if (patch.num_epochs !== undefined) {
      setEpochs(String(patch.num_epochs))
    }

    if (
      patch.start_from_sft_checkpoint &&
      supportsInitAdapter &&
      !initAdapterPath.trim()
    ) {
      toast.info(describeInitialAdapterRecipeHint(INITIAL_ADAPTER_LABEL))
    } else {
      toast.success('Applied the safe preference starting recipe')
    }
  }

  function handleSubmit() {
    if (!modelPath || !datasetPath) {
      toast.error('Model path and dataset path are required')
      return
    }

    if (!selectedTechniqueOption?.enabled) {
      toast.error(selectedTechniqueOption?.reason ?? 'This training path is not available')
      return
    }

    const parsedEpochs = parseInt(epochs, 10)
    const parsedBatchSize = parseInt(batchSize, 10)
    const parsedLoraR = parseInt(loraR, 10)
    const parsedLoraAlpha = parseInt(loraAlpha, 10)
    const parsedLoraDropout = parseFloat(loraDropout)
    const parsedWarmupRatio = parseFloat(warmupRatio)
    const parsedWeightDecay = parseFloat(weightDecay)
    const parsedGradAccum = parseInt(gradAccum, 10)
    const parsedMaxSeqLength = parseInt(maxSeqLength, 10)
    const parsedLearningRate = parseFloat(learningRate)
    const parsedNumStages = parseInt(numStages, 10)
    const resolvedOutputDir = outputDir.trim() || buildDefaultOutputDir(technique, sequential, datasetPath)
    const resolvedUseLora = defaultsToLoraTraining(technique)

    if (supportsInitAdapter && initAdapterPath.trim() && !resolvedUseLora) {
      toast.error(describeInitialAdapterRequirement())
      return
    }

    const commonArgs: Record<string, unknown> = {
      output_dir: resolvedOutputDir,
      base_model: modelPath.trim(),
      dataset_path: datasetPath.trim(),
      load_in_4bit: quantization === '4bit',
      ...(supportsInitAdapter && initAdapterPath.trim() ? { adapter_path: initAdapterPath.trim() } : {}),
      ...(trainingRecipe !== 'none' ? { recipe: trainingRecipe } : {}),
    }

    let args: Record<string, unknown> = { ...commonArgs }

    if (technique === 'sft' || technique === 'vlm_sft') {
      args = {
        ...commonArgs,
        num_epochs: parsedEpochs,
        learning_rate: parsedLearningRate,
        per_device_train_batch_size: parsedBatchSize,
        lora_r: parsedLoraR,
        lora_alpha: parsedLoraAlpha,
        lora_dropout: parsedLoraDropout,
        warmup_ratio: parsedWarmupRatio,
        weight_decay: parsedWeightDecay,
        gradient_accumulation_steps: parsedGradAccum,
        max_seq_length: parsedMaxSeqLength,
        ...(showEvalDatasetField && evalDatasetPath.trim() ? { eval_file_path: evalDatasetPath.trim() } : {}),
      }
    } else if (technique === 'curriculum') {
      args = {
        ...commonArgs,
        num_stages: parsedNumStages,
        num_epochs_per_stage: parsedEpochs,
        score_column: scoreColumn.trim() || 'weighted_score',
        difficulty_order: difficultyOrder,
        use_lora: true,
        lora_r: parsedLoraR,
        lora_alpha: parsedLoraAlpha,
      }
    } else if (technique === 'dpo' || technique === 'kto') {
      args = {
        ...commonArgs,
        num_epochs: parsedEpochs,
        use_lora: true,
        lora_r: parsedLoraR,
        lora_alpha: parsedLoraAlpha,
        lora_dropout: parsedLoraDropout,
      }
    } else if (technique === 'grpo') {
      args = {
        ...commonArgs,
        num_epochs: parsedEpochs,
        use_lora: true,
        lora_r: parsedLoraR,
        lora_alpha: parsedLoraAlpha,
        lora_dropout: parsedLoraDropout,
        learning_rate: parsedLearningRate,
        per_device_train_batch_size: parsedBatchSize,
        gradient_accumulation_steps: parsedGradAccum,
      }
    } else {
      args = {
        ...commonArgs,
        num_epochs: parsedEpochs,
      }
    }

    if (sequential) {
      const sequentialStage = {
        technique,
        dataset_path: datasetPath.trim(),
        num_epochs: parsedEpochs,
        ...(trainingRecipe !== 'none' ? { recipe: trainingRecipe } : {}),
        ...(technique === 'sft'
          ? {
              learning_rate: parsedLearningRate,
              per_device_train_batch_size: parsedBatchSize,
              lora_r: parsedLoraR,
              lora_alpha: parsedLoraAlpha,
              lora_dropout: parsedLoraDropout,
              warmup_ratio: parsedWarmupRatio,
              weight_decay: parsedWeightDecay,
              gradient_accumulation_steps: parsedGradAccum,
              max_seq_length: parsedMaxSeqLength,
            }
          : {}),
        ...((technique === 'dpo' || technique === 'kto')
          ? {
              use_lora: true,
              lora_r: parsedLoraR,
              lora_alpha: parsedLoraAlpha,
              lora_dropout: parsedLoraDropout,
              ...(trainingRecipe !== 'none' ? { recipe: trainingRecipe } : {}),
            }
          : {}),
        ...(technique === 'grpo'
          ? {
              use_lora: true,
              lora_r: parsedLoraR,
              lora_alpha: parsedLoraAlpha,
              lora_dropout: parsedLoraDropout,
              learning_rate: parsedLearningRate,
              per_device_train_batch_size: parsedBatchSize,
              gradient_accumulation_steps: parsedGradAccum,
            }
          : {}),
        load_in_4bit: quantization === '4bit',
      }

      args = {
        output_dir: resolvedOutputDir,
        base_model: modelPath.trim(),
        stages: JSON.stringify([sequentialStage]),
      }
    }

    if (supportsBenchmarkWorkflow && benchmarkAfterTraining) {
      const benchmarkDatasetPath = evalDatasetPath.trim()
      if (!benchmarkDatasetPath) {
        toast.error('Validation dataset is required when benchmark-after-training is enabled')
        return
      }

      const normalizedOutputDir = resolvedOutputDir.replace(/[\\/]+$/, '')
      const benchmarkOutputDir = normalizedOutputDir.endsWith('-benchmark')
        ? normalizedOutputDir
        : `${normalizedOutputDir}-benchmark`
      const benchmarkSeeds = optimizeAcrossRuns ? [3407, 42, 1234] : [3407]
      const steps = [
        {
          tool: 'workflow.benchmark_finetuning',
          params: {
            train_dataset_path: datasetPath.trim(),
            output_dir: benchmarkOutputDir,
            base_model: modelPath.trim(),
            eval_file_path: benchmarkDatasetPath,
            dev_data_path: benchmarkDatasetPath,
            include_flat_sft: true,
            include_curriculum_sft: false,
            seeds: benchmarkSeeds,
            primary_pack: 'dev',
            num_epochs_flat: parsedEpochs,
            use_lora: true,
            lora_r: parsedLoraR,
            lora_alpha: parsedLoraAlpha,
            lora_dropout: parsedLoraDropout,
            load_in_4bit: quantization === '4bit',
            learning_rate: parsedLearningRate,
            per_device_train_batch_size: parsedBatchSize,
            gradient_accumulation_steps: parsedGradAccum,
            max_seq_length: parsedMaxSeqLength,
            warmup_ratio: parsedWarmupRatio,
            weight_decay: parsedWeightDecay,
            save_best_model: true,
            ...(trainingRecipe !== 'none' ? { recipe: trainingRecipe } : {}),
          },
        },
      ]

      runPipeline.mutate(
        { steps: JSON.stringify(steps) },
        {
          onSuccess: () => {
            toast.success(
              optimizeAcrossRuns
                ? 'Optimization workflow started'
                : 'Training benchmark workflow started',
            )
            setOutputDir(buildDefaultOutputDir(technique, sequential, datasetPath))
            setOutputDirCustomized(false)
            onSubmit()
          },
          onError: (error) => {
            toast.error(`Failed to start workflow: ${error.message}`)
          },
        },
      )
      return
    }

    startTraining.mutate(
      { technique: sequential ? 'sequential' : technique, args },
      {
        onSuccess: () => {
          toast.success('Training job started')
          setOutputDir(buildDefaultOutputDir(technique, sequential, datasetPath))
          setOutputDirCustomized(false)
          onSubmit()
        },
        onError: (error) => {
          if (/timed out/i.test(error.message)) {
            toast.warning('Training start is taking longer than expected. Checking the backend job list now.')
            return
          }
          toast.error(`Failed to start training: ${error.message}`)
        },
      },
    )
  }

  if (!open) return null

  return (
    <Card>
      <CardHeader className="pb-4">
        <CardTitle className="text-base">New Training Job</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <TrainingTechniqueSelector
          options={techniqueOptions}
          value={technique}
          onChange={setTechnique}
        />

        <div className="space-y-2">
          <label className="text-sm font-medium">Training Recipe</label>
          <select
            value={trainingRecipe}
            onChange={(event) => setTrainingRecipe(event.target.value as (typeof TRAINING_RECIPE_OPTIONS)[number]['value'])}
            className="w-full h-9 rounded-md border border-input bg-transparent px-3 text-sm text-foreground"
          >
            {TRAINING_RECIPE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-muted-foreground">
            {TRAINING_RECIPE_OPTIONS.find((option) => option.value === trainingRecipe)?.helper}
          </p>
        </div>

        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={sequential}
            onChange={(event) => setSequential(event.target.checked)}
            className="rounded border-input"
            disabled={!sequentialAllowed}
          />
          Sequential Training
        </label>
        {!sequentialAllowed && technique === 'curriculum' && (
          <p className="text-xs text-muted-foreground">
            Curriculum already runs stage-by-stage, so sequential chaining is disabled here.
          </p>
        )}
        {!sequentialAllowed && technique === 'vlm_sft' && (
          <p className="text-xs text-muted-foreground">
            Sequential chaining stays off for VLM until the backend exposes multimodal multi-stage trainers.
          </p>
        )}

        {vlmSupportMissing && (
          <div className="rounded-md border border-amber-300/40 bg-amber-500/5 px-3 py-2 text-xs text-muted-foreground">
            The selected model looks like a vision-language model. Training stays disabled until the gateway exposes a dedicated VLM SFT endpoint.
          </div>
        )}

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Model</label>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-7 gap-1 text-xs"
              disabled={autoSuggestDisabled}
              onClick={() => {
                autoSuggest.mutate(
                  {
                    dataset_path: datasetPath,
                    technique: validationTechnique ?? 'sft',
                    use_case: 'general',
                  },
                  {
                    onSuccess: (result) => {
                      if (result.technique_warning) {
                        toast.warning(result.technique_warning)
                      }
                      if (result.success && result.candidates.length > 0) {
                        setSuggestions(result.candidates.slice(0, 3))
                      } else {
                        toast.error(result.error || 'No models fit your hardware')
                        setSuggestions([])
                      }
                    },
                    onError: (error) => toast.error(error.message),
                  },
                )
              }}
            >
              <Sparkles className="h-3 w-3" />
              {autoSuggest.isPending ? 'Analyzing...' : 'Suggest Model'}
            </Button>
          </div>
          <ModelBrowser value={modelPath} onChange={onModelPathChange} />
          {modelPath && (
            <p className="text-xs text-muted-foreground">
              {modelModality === 'vision-language'
                ? 'Detected modality: vision-language'
                : modelModality === 'text'
                  ? 'Detected modality: text'
                  : 'Detected modality: unknown'}
            </p>
          )}
          {suggestions.length > 0 && (
            <div className="space-y-1 rounded-md border border-border p-2">
              <p className="text-xs font-medium text-muted-foreground">Suggested models</p>
              {suggestions.map((candidate) => (
                <button
                  key={candidate.model_id}
                  type="button"
                  className="w-full cursor-pointer rounded px-2 py-1.5 text-left hover:bg-accent transition-colors"
                  onClick={() => applySuggestedConfig(candidate)}
                >
                  <div className="text-sm font-medium">{candidate.model_id}</div>
                  <div className="text-xs text-muted-foreground">{candidate.why_recommended}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        {supportsInitAdapter && (
          <div className="space-y-2">
            <label className="text-sm font-medium">{INITIAL_ADAPTER_LABEL}</label>
            <ModelPathField
              value={initAdapterPath}
              onChange={setInitAdapterPath}
              disabled={isSubmitting}
              validationPurpose="adapter"
              placeholder="./output/best_sft_adapter"
              helperText={describeContinueFromAdapter()}
            />
          </div>
        )}

        <TrainingDatasetField
          label="Train Dataset"
          datasetPath={datasetPath}
          onChange={setDatasetPath}
          datasets={datasets}
          schemaValid={schemaValid}
          qualityValid={qualityValid}
          placeholder={datasetPlaceholder}
          hint={
            validationTechnique
              ? datasetHelpText
              : 'Schema validation for this training mode is not advertised by the current backend yet.'
          }
        />

        {showPreferenceDatasetAnalysis && (
          <PreferenceDatasetAnalysisCard
            datasetPath={datasetPath}
            technique={preferenceTechnique}
            onApplyStartingRecipe={applyPreferenceStartingRecipe}
          />
        )}

        {showEvalDatasetField && (
          <TrainingDatasetField
            label="Validation Dataset (optional)"
            datasetPath={evalDatasetPath}
            onChange={setEvalDatasetPath}
            datasets={datasets}
            schemaValid={null}
            qualityValid={null}
            placeholder="/path/to/eval.jsonl"
            hint="Optional. When set, the trainer will evaluate during SFT and can save the best checkpoint instead of training blind."
          />
        )}

        {supportsBenchmarkWorkflow && (
          <div className="space-y-3 rounded-md border border-border/60 bg-secondary/20 p-3">
            <div>
              <p className="text-sm font-medium">Post-training evaluation</p>
              <p className="text-xs text-muted-foreground">
                Uses the Validation Dataset as the dev evaluation pack when benchmarking is enabled.
              </p>
            </div>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={benchmarkAfterTraining}
                onChange={(event) => {
                  const checked = event.target.checked
                  setBenchmarkAfterTraining(checked)
                  if (!checked) {
                    setOptimizeAcrossRuns(false)
                  }
                }}
                className="rounded border-input"
              />
              Benchmark after training
            </label>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={optimizeAcrossRuns}
                onChange={(event) => {
                  const checked = event.target.checked
                  setOptimizeAcrossRuns(checked)
                  if (checked) {
                    setBenchmarkAfterTraining(true)
                  }
                }}
                className="rounded border-input"
                disabled={!benchmarkAfterTraining}
              />
              Optimize across multiple runs
            </label>
            {benchmarkAfterTraining && (
              <p className="text-xs text-muted-foreground">
                Single-run benchmark uses seed <code>3407</code>. Optimization runs the default seed set <code>3407, 42, 1234</code> and keeps the best-scoring candidate in the benchmark report.
              </p>
            )}
            {benchmarkAfterTraining && !evalDatasetPath.trim() && (
              <p className="text-xs text-amber-300">
                Set Validation Dataset to enable benchmark scoring and best-checkpoint selection.
              </p>
            )}
          </div>
        )}

        <div className="space-y-2">
          <label className="text-sm font-medium">Hyperparameters</label>
          <div className="grid grid-cols-2 gap-3">
            {technique === 'curriculum' ? (
              <>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">num_stages</label>
                  <Input type="number" min="2" value={numStages} onChange={(event) => setNumStages(event.target.value)} />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">epochs_per_stage</label>
                  <Input type="number" value={epochs} onChange={(event) => setEpochs(event.target.value)} />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">difficulty_order</label>
                  <select
                    value={difficultyOrder}
                    onChange={(event) => setDifficultyOrder(event.target.value === 'hard_first' ? 'hard_first' : 'easy_first')}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    <option value="easy_first">easy_first</option>
                    <option value="hard_first">hard_first</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">score_column</label>
                  <Input value={scoreColumn} onChange={(event) => setScoreColumn(event.target.value)} />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">lora_r</label>
                  <Input type="number" value={loraR} onChange={(event) => setLoraR(event.target.value)} />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">lora_alpha</label>
                  <Input type="number" value={loraAlpha} onChange={(event) => setLoraAlpha(event.target.value)} />
                </div>
              </>
            ) : (
              <>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">learning_rate</label>
                  <Input value={learningRate} onChange={(event) => setLearningRate(event.target.value)} />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">epochs</label>
                  <Input type="number" value={epochs} onChange={(event) => setEpochs(event.target.value)} />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">batch_size</label>
                  <Input type="number" value={batchSize} onChange={(event) => setBatchSize(event.target.value)} />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">lora_r</label>
                  <Input type="number" value={loraR} onChange={(event) => setLoraR(event.target.value)} />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">lora_alpha</label>
                  <Input type="number" value={loraAlpha} onChange={(event) => setLoraAlpha(event.target.value)} />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">lora_dropout</label>
                  <Input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={loraDropout}
                    onChange={(event) => setLoraDropout(event.target.value)}
                  />
                </div>
              </>
            )}
          </div>
          {technique === 'curriculum' && (
            <p className="text-xs text-muted-foreground">
              Leave `score_column` as `weighted_score` for pre-scored datasets, or point it to another field such as `complexity`. If that column is missing, the backend will try to auto-score the dataset through the evaluator pipeline, which requires the evaluator stack and provider credentials.
            </p>
          )}
          {technique === 'vlm_sft' && (
            <p className="text-xs text-muted-foreground">
              VLM SFT uses the same core training knobs as text SFT, but expects a multimodal dataset manifest instead of a plain text dataset.
            </p>
          )}
        </div>

        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced((current) => !current)}
            className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
          >
            <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', showAdvanced && 'rotate-180')} />
            Advanced
          </button>
          {showAdvanced && (
            <div className="mt-3 grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">output_dir</label>
                <Input
                  value={outputDir}
                  onChange={(event) => {
                    setOutputDir(event.target.value)
                    setOutputDirCustomized(true)
                  }}
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">quantization</label>
                <select
                  value={quantization}
                  onChange={(event) => setQuantization(event.target.value === 'none' ? 'none' : '4bit')}
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  <option value="4bit">4-bit (recommended, saves memory)</option>
                  <option value="none">None (full precision)</option>
                </select>
              </div>
              {technique !== 'curriculum' && (
                <>
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">warmup_ratio</label>
                    <Input value={warmupRatio} onChange={(event) => setWarmupRatio(event.target.value)} />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">weight_decay</label>
                    <Input value={weightDecay} onChange={(event) => setWeightDecay(event.target.value)} />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">gradient_accumulation_steps</label>
                    <Input type="number" value={gradAccum} onChange={(event) => setGradAccum(event.target.value)} />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">max_seq_length</label>
                    <Input type="number" value={maxSeqLength} onChange={(event) => setMaxSeqLength(event.target.value)} />
                  </div>
                </>
              )}
              <p className="col-span-2 text-xs text-muted-foreground">
                {technique === 'curriculum'
                  ? describeCurriculumAdvancedControls()
                  : technique === 'vlm_sft'
                    ? 'Keep VLM settings close to text SFT defaults unless the backend exposes stronger modality-specific guidance.'
                    : '4-bit loading reduces memory usage during training. Use full precision only if you have enough VRAM/RAM.'}
              </p>
            </div>
          )}
        </div>

        <Button onClick={handleSubmit} disabled={!canSubmit} className="w-full">
          {submitLabel}
        </Button>
      </CardContent>
    </Card>
  )
}
