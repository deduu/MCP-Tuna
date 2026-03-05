import { useState, useEffect, useCallback, useRef } from 'react'
import { toast } from 'sonner'
import { useStartTraining } from '@/api/hooks/useTraining'
import { useDatasets } from '@/api/hooks/useDatasets'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { ChevronDown } from 'lucide-react'
import { ModelBrowser } from './ModelBrowser'

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

  // Hyperparameters
  const [learningRate, setLearningRate] = useState('2e-4')
  const [epochs, setEpochs] = useState('3')
  const [batchSize, setBatchSize] = useState('4')
  const [loraR, setLoraR] = useState('16')
  const [loraAlpha, setLoraAlpha] = useState('32')

  // Advanced
  const [warmupSteps, setWarmupSteps] = useState('0')
  const [weightDecay, setWeightDecay] = useState('0.01')
  const [gradAccum, setGradAccum] = useState('1')
  const [maxSeqLength, setMaxSeqLength] = useState('2048')

  // Validation badges
  const [schemaValid, setSchemaValid] = useState<'pass' | 'warn' | null>(null)
  const [qualityValid, setQualityValid] = useState<'pass' | 'warn' | null>(null)

  const { data: datasets = [] } = useDatasets()
  const startTraining = useStartTraining()
  const validateTool = useToolExecution()
  const qualityTool = useToolExecution()

  // Debounced pre-validation
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const runValidation = useCallback(() => {
    if (!datasetPath) {
      setSchemaValid(null)
      setQualityValid(null)
      return
    }

    validateTool.mutate(
      { toolName: 'validate.schema', args: { dataset_path: datasetPath, technique } },
      {
        onSuccess: (res) => setSchemaValid(res.success ? 'pass' : 'warn'),
        onError: () => setSchemaValid('warn'),
      },
    )
    qualityTool.mutate(
      { toolName: 'validate.data_quality', args: { dataset_path: datasetPath } },
      {
        onSuccess: (res) => setQualityValid(res.success ? 'pass' : 'warn'),
        onError: () => setQualityValid('warn'),
      },
    )
  }, [datasetPath, technique])

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(runValidation, 800)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [runValidation])

  function handleSubmit() {
    if (!modelPath || !datasetPath) {
      toast.error('Model path and dataset path are required')
      return
    }

    const args: Record<string, unknown> = {
      model_name: modelPath,
      dataset_path: datasetPath,
      learning_rate: parseFloat(learningRate),
      num_train_epochs: parseInt(epochs),
      per_device_train_batch_size: parseInt(batchSize),
      lora_r: parseInt(loraR),
      lora_alpha: parseInt(loraAlpha),
      warmup_steps: parseInt(warmupSteps),
      weight_decay: parseFloat(weightDecay),
      gradient_accumulation_steps: parseInt(gradAccum),
      max_seq_length: parseInt(maxSeqLength),
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
          <label className="text-sm font-medium">Model</label>
          <ModelBrowser value={modelPath} onChange={setModelPath} />
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
              placeholder="Dataset path..."
            />
            {datasets.length > 0 && (
              <select
                className="absolute right-1 top-1 h-7 rounded border-none bg-transparent text-xs text-muted-foreground cursor-pointer"
                value=""
                onChange={(e) => {
                  if (e.target.value) setDatasetPath(e.target.value)
                }}
              >
                <option value="">Select...</option>
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
                <label className="text-xs text-muted-foreground">warmup_steps</label>
                <Input type="number" value={warmupSteps} onChange={(e) => setWarmupSteps(e.target.value)} />
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
