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
import { useDatasets } from '@/api/hooks/useDatasets'
import { mcpCall } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  buildDefaultOutputDir,
  getDatasetHelpText,
  getDatasetPlaceholder,
  getTechniqueOptions,
  inferModelModality,
  resolveValidationTechnique,
  supportsSequentialTraining,
} from '@/lib/training-capabilities'
import { cn } from '@/lib/utils'
import { ModelBrowser } from './ModelBrowser'
import { TrainingDatasetField } from './TrainingDatasetField'
import { TrainingTechniqueSelector } from './TrainingTechniqueSelector'

interface NewTrainingPanelProps {
  open: boolean
  onToggle: () => void
  onSubmit: () => void
  modelPath: string
  onModelPathChange: (value: string) => void
}

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
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [outputDir, setOutputDir] = useState(() => buildDefaultOutputDir('sft', false))
  const [outputDirCustomized, setOutputDirCustomized] = useState(false)
  const [quantization, setQuantization] = useState<'4bit' | 'none'>('4bit')

  const [learningRate, setLearningRate] = useState('2e-4')
  const [epochs, setEpochs] = useState('3')
  const [batchSize, setBatchSize] = useState('4')
  const [loraR, setLoraR] = useState('16')
  const [loraAlpha, setLoraAlpha] = useState('32')

  const [warmupRatio, setWarmupRatio] = useState('0')
  const [weightDecay, setWeightDecay] = useState('0.01')
  const [gradAccum, setGradAccum] = useState('1')
  const [maxSeqLength, setMaxSeqLength] = useState('2048')
  const [numStages, setNumStages] = useState('3')
  const [scoreColumn, setScoreColumn] = useState('weighted_score')
  const [difficultyOrder, setDifficultyOrder] = useState<'easy_first' | 'hard_first'>('easy_first')

  const [schemaValid, setSchemaValid] = useState<'pass' | 'warn' | null>(null)
  const [qualityValid, setQualityValid] = useState<'pass' | 'warn' | null>(null)
  const [suggestions, setSuggestions] = useState<AutoPrescribeCandidate[]>([])

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const { data: datasets = [] } = useDatasets()
  const { data: localCandidates = [] } = useLocalModelCandidates()
  const { data: trainingCapabilities } = useTrainingCapabilities()
  const startTraining = useStartTraining()
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
  const validationTechnique = resolveValidationTechnique(technique, trainingCapabilities)
  const datasetPlaceholder = getDatasetPlaceholder(technique)
  const datasetHelpText = getDatasetHelpText(technique)
  const vlmSupportMissing = modelModality === 'vision-language' && !trainingCapabilities?.supports_vlm_sft
  const canSubmit = Boolean(modelPath && datasetPath && selectedTechniqueOption?.enabled && !startTraining.isPending)
  const submitLabel = startTraining.isPending ? 'Starting...' : 'Start Training'
  const autoSuggestDisabled = !datasetPath || autoSuggest.isPending || technique === 'vlm_sft'
  const showEvalDatasetField = technique === 'sft' && !sequential
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
      setOutputDir(buildDefaultOutputDir(technique, sequential))
    }
  }, [technique, sequential, outputDirCustomized])

  useEffect(() => {
    if (!sequentialAllowed && sequential) {
      setSequential(false)
    }
  }, [sequential, sequentialAllowed])

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
    const parsedWarmupRatio = parseFloat(warmupRatio)
    const parsedWeightDecay = parseFloat(weightDecay)
    const parsedGradAccum = parseInt(gradAccum, 10)
    const parsedMaxSeqLength = parseInt(maxSeqLength, 10)
    const parsedLearningRate = parseFloat(learningRate)
    const parsedNumStages = parseInt(numStages, 10)
    const resolvedOutputDir = outputDir.trim() || buildDefaultOutputDir(technique, sequential)

    const commonArgs: Record<string, unknown> = {
      output_dir: resolvedOutputDir,
      base_model: modelPath.trim(),
      dataset_path: datasetPath.trim(),
      load_in_4bit: quantization === '4bit',
    }

    let args: Record<string, unknown> = { ...commonArgs }

    if (technique === 'sft' || technique === 'vlm_sft') {
      args = {
        ...commonArgs,
        learning_rate: parsedLearningRate,
        per_device_train_batch_size: parsedBatchSize,
        lora_r: parsedLoraR,
        lora_alpha: parsedLoraAlpha,
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
        ...(technique === 'sft'
          ? {
              learning_rate: parsedLearningRate,
              per_device_train_batch_size: parsedBatchSize,
              lora_r: parsedLoraR,
              lora_alpha: parsedLoraAlpha,
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

    startTraining.mutate(
      { technique: sequential ? 'sequential' : technique, args },
      {
        onSuccess: () => {
          toast.success('Training job started')
          setOutputDir(buildDefaultOutputDir(technique, sequential))
          setOutputDirCustomized(false)
          onSubmit()
        },
        onError: (error) => {
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
                  ? 'Curriculum async training currently exposes stage and scoring controls, plus LoRA and quantization.'
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
