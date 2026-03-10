import { useState, useEffect, useRef } from 'react'
import { toast } from 'sonner'
import { useStartTraining, useAutoSuggestModel } from '@/api/hooks/useTraining'
import { useDatasets } from '@/api/hooks/useDatasets'
import { mcpCall } from '@/api/client'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { ChevronDown, Sparkles } from 'lucide-react'
import { ModelBrowser } from './ModelBrowser'
import type { AutoPrescribeCandidate } from '@/api/types'

interface NewTrainingPanelProps {
  open: boolean
  onToggle: () => void
  onSubmit: () => void
}

type Technique = 'sft' | 'dpo' | 'grpo' | 'kto'

const TECHNIQUES: { value: Technique; label: string }[] = [
  { value: 'sft', label: 'SFT' },
  { value: 'dpo', label: 'DPO' },
  { value: 'grpo', label: 'GRPO' },
  { value: 'kto', label: 'KTO' },
]

export function NewTrainingPanel({ open, onToggle: _onToggle, onSubmit }: NewTrainingPanelProps) {
  const [technique, setTechnique] = useState<Technique>('sft')
  const [sequential, setSequential] = useState(false)
  const [modelPath, setModelPath] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [outputDir, setOutputDir] = useState('./output')
  const [quantization, setQuantization] = useState<'4bit' | 'none'>('4bit')

  // Hyperparameters
  const [learningRate, setLearningRate] = useState('2e-4')
  const [epochs, setEpochs] = useState('3')
  const [batchSize, setBatchSize] = useState('4')
  const [loraR, setLoraR] = useState('16')
  const [loraAlpha, setLoraAlpha] = useState('32')

  // Advanced
  const [warmupRatio, setWarmupRatio] = useState('0')
  const [weightDecay, setWeightDecay] = useState('0.01')
  const [gradAccum, setGradAccum] = useState('1')
  const [maxSeqLength, setMaxSeqLength] = useState('2048')

  // Validation badges
  const [schemaValid, setSchemaValid] = useState<'pass' | 'warn' | null>(null)
  const [qualityValid, setQualityValid] = useState<'pass' | 'warn' | null>(null)

  const [suggestions, setSuggestions] = useState<AutoPrescribeCandidate[]>([])

  const { data: datasets = [] } = useDatasets()
  const startTraining = useStartTraining()
  const autoSuggest = useAutoSuggestModel()

  // Debounced pre-validation — uses mcpCall directly to avoid mutation state issues
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!datasetPath) {
      setSchemaValid(null)
      setQualityValid(null)
      return
    }

    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      mcpCall<{ success: boolean; technique_detected?: string; missing_columns?: string[] }>(
        'validate.schema', { dataset_path: datasetPath, technique },
      )
        .then((res) => {
          setSchemaValid(res.success ? 'pass' : 'warn')
          if (!res.success && (res.missing_columns ?? []).length > 0) {
            const detected = res.technique_detected
              ? `${res.technique_detected.toUpperCase()} format`
              : 'unknown format'
            toast.warning(
              `Dataset is ${detected} — missing ${(res.missing_columns ?? []).join(', ')} for ${technique.toUpperCase()}`,
            )
          }
        })
        .catch(() => setSchemaValid('warn'))

      mcpCall<{ success: boolean }>('validate.data_quality', { dataset_path: datasetPath })
        .then((res) => setQualityValid(res.success ? 'pass' : 'warn'))
        .catch(() => setQualityValid('warn'))
    }, 800)

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [datasetPath, technique])

  function handleSubmit() {
    if (!modelPath || !datasetPath) {
      toast.error('Model path and dataset path are required')
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

    const commonArgs: Record<string, unknown> = {
      output_dir: outputDir.trim() || './output',
      base_model: modelPath.trim(),
      dataset_path: datasetPath.trim(),
      num_epochs: parsedEpochs,
      load_in_4bit: quantization === '4bit',
    }

    let args: Record<string, unknown> = { ...commonArgs }

    if (technique === 'sft') {
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
      }
    } else if (technique === 'dpo') {
      args = {
        ...commonArgs,
        use_lora: true,
        lora_r: parsedLoraR,
      }
    } else if (technique === 'kto') {
      args = {
        ...commonArgs,
        use_lora: true,
        lora_r: parsedLoraR,
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
        output_dir: outputDir.trim() || './output',
        base_model: modelPath.trim(),
        stages: JSON.stringify([sequentialStage]),
      }
    }

    const selectedTechnique = sequential ? 'sequential' : technique

    startTraining.mutate(
      { technique: selectedTechnique, args },
      {
        onSuccess: () => {
          toast.success('Training job started')
          onSubmit()
        },
        onError: (err) => {
          toast.error(`Failed to start training: ${err.message}`)
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
        {/* Technique selector */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Technique</label>
          <div className="flex rounded-lg border border-border overflow-hidden">
            {TECHNIQUES.map((t) => (
              <button
                key={t.value}
                type="button"
                onClick={() => setTechnique(t.value)}
                className={cn(
                  'flex-1 cursor-pointer px-3 py-2 text-sm font-medium transition-colors',
                  technique === t.value
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-transparent text-muted-foreground hover:bg-accent',
                )}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* Sequential toggle */}
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={sequential}
            onChange={(e) => setSequential(e.target.checked)}
            className="rounded border-input"
          />
          Sequential Training
        </label>

        {/* Model path */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Model</label>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-7 gap-1 text-xs"
              disabled={!datasetPath || autoSuggest.isPending}
              onClick={() => {
                autoSuggest.mutate(
                  { dataset_path: datasetPath, technique, use_case: 'general' },
                  {
                    onSuccess: (data) => {
                      if (data.technique_warning) {
                        toast.warning(data.technique_warning)
                      }
                      if (data.success && data.candidates.length > 0) {
                        setSuggestions(data.candidates.slice(0, 3))
                      } else {
                        toast.error(data.error || 'No models fit your hardware')
                        setSuggestions([])
                      }
                    },
                    onError: (err) => toast.error(err.message),
                  },
                )
              }}
            >
              <Sparkles className="h-3 w-3" />
              {autoSuggest.isPending ? 'Analyzing...' : 'Suggest Model'}
            </Button>
          </div>
          <ModelBrowser value={modelPath} onChange={setModelPath} />
          {suggestions.length > 0 && (
            <div className="space-y-1 rounded-md border border-border p-2">
              <p className="text-xs font-medium text-muted-foreground">Suggested models</p>
              {suggestions.map((s) => (
                <button
                  key={s.model_id}
                  type="button"
                  className="w-full cursor-pointer rounded px-2 py-1.5 text-left hover:bg-accent transition-colors"
                  onClick={() => {
                    setModelPath(s.model_id)
                    const cfg = s.prescribe_config?.config ?? {}
                    if (cfg.learning_rate) setLearningRate(String(cfg.learning_rate))
                    if (cfg.num_epochs) setEpochs(String(cfg.num_epochs))
                    if (cfg.per_device_train_batch_size) setBatchSize(String(cfg.per_device_train_batch_size))
                    if (cfg.lora_r) setLoraR(String(cfg.lora_r))
                    if (cfg.lora_alpha) setLoraAlpha(String(cfg.lora_alpha))
                    if (cfg.gradient_accumulation_steps) setGradAccum(String(cfg.gradient_accumulation_steps))
                    if (cfg.max_seq_length) setMaxSeqLength(String(cfg.max_seq_length))
                    setSuggestions([])
                    toast.success(`Applied ${s.model_id} with optimized config`)
                  }}
                >
                  <div className="text-sm font-medium">{s.model_id}</div>
                  <div className="text-xs text-muted-foreground">{s.why_recommended}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Dataset path */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">Dataset</label>
            {schemaValid && (
              <Badge variant={schemaValid === 'pass' ? 'success' : 'warning'} className="text-[10px] px-1.5 py-0">
                schema {schemaValid}
              </Badge>
            )}
            {qualityValid && (
              <Badge variant={qualityValid === 'pass' ? 'success' : 'warning'} className="text-[10px] px-1.5 py-0">
                quality {qualityValid}
              </Badge>
            )}
          </div>
          <div className="relative">
            <Input
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              placeholder="Path to .jsonl, .json, .csv, or .parquet file..."
            />
            {datasets.length > 0 && (
              <select
                className="absolute right-1 top-1/2 -translate-y-1/2 h-7 w-7 appearance-none rounded border-none bg-transparent text-xs text-muted-foreground cursor-pointer opacity-60 hover:opacity-100"
                value=""
                title="Pick a dataset"
                onChange={(e) => {
                  if (e.target.value) setDatasetPath(e.target.value)
                }}
              >
                <option value="">▾</option>
                {datasets.map((ds) => (
                  <option key={ds.dataset_id} value={ds.file_path}>
                    {ds.file_path} ({ds.row_count} rows)
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>

        {/* Hyperparameters grid */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Hyperparameters</label>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">learning_rate</label>
              <Input value={learningRate} onChange={(e) => setLearningRate(e.target.value)} />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">epochs</label>
              <Input type="number" value={epochs} onChange={(e) => setEpochs(e.target.value)} />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">batch_size</label>
              <Input type="number" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">lora_r</label>
              <Input type="number" value={loraR} onChange={(e) => setLoraR(e.target.value)} />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">lora_alpha</label>
              <Input type="number" value={loraAlpha} onChange={(e) => setLoraAlpha(e.target.value)} />
            </div>
          </div>
        </div>

        {/* Advanced section */}
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced((a) => !a)}
            className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
          >
            <ChevronDown
              className={cn('h-3.5 w-3.5 transition-transform', showAdvanced && 'rotate-180')}
            />
            Advanced
          </button>
          {showAdvanced && (
            <div className="grid grid-cols-2 gap-3 mt-3">
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">output_dir</label>
                <Input value={outputDir} onChange={(e) => setOutputDir(e.target.value)} />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">quantization</label>
                <select
                  value={quantization}
                  onChange={(e) => setQuantization(e.target.value === 'none' ? 'none' : '4bit')}
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  <option value="4bit">4-bit (recommended, saves memory)</option>
                  <option value="none">None (full precision)</option>
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">warmup_ratio</label>
                <Input value={warmupRatio} onChange={(e) => setWarmupRatio(e.target.value)} />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">weight_decay</label>
                <Input value={weightDecay} onChange={(e) => setWeightDecay(e.target.value)} />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">gradient_accumulation_steps</label>
                <Input type="number" value={gradAccum} onChange={(e) => setGradAccum(e.target.value)} />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">max_seq_length</label>
                <Input type="number" value={maxSeqLength} onChange={(e) => setMaxSeqLength(e.target.value)} />
              </div>
              <p className="col-span-2 text-xs text-muted-foreground">
                4-bit loading reduces memory usage during training. Use full precision only if you have enough VRAM/RAM.
              </p>
            </div>
          )}
        </div>

        {/* Submit */}
        <Button
          onClick={handleSubmit}
          disabled={startTraining.isPending || !modelPath || !datasetPath}
          className="w-full"
        >
          {startTraining.isPending ? 'Starting...' : 'Start Training'}
        </Button>
      </CardContent>
    </Card>
  )
}
