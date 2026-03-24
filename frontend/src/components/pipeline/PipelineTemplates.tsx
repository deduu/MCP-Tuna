import { useEffect, useMemo, useState } from 'react'
import { Play, Settings2, ChevronDown } from 'lucide-react'
import { useRunFullPipeline, useRunPipeline } from '@/api/hooks/usePipeline'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { CustomPipelineForm } from './CustomPipelineForm'
import { DocumentPathInput } from './DocumentPathInput'
import { DocumentPathListInput } from './DocumentPathListInput'
import { ModelPathField } from './ModelPathField'
import { buildPipelineOutputDir } from '@/lib/training-capabilities'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'

const TECHNIQUES = ['sft', 'dpo', 'grpo', 'kto'] as const

function parseDocumentPaths(value: string): string[] {
  return value
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter(Boolean)
}

export function PipelineTemplates() {
  const [singleDocPath, setSingleDocPath] = useState('')
  const [multiDocPaths, setMultiDocPaths] = useState('')
  const [useMultipleDocuments, setUseMultipleDocuments] = useState(false)
  const [technique, setTechnique] = useState<string>('sft')
  const [modelPath, setModelPath] = useState('')
  const [lastResult, setLastResult] = useState<Record<string, unknown> | null>(null)
  const [showFullAdvanced, setShowFullAdvanced] = useState(false)
  const [outputDir, setOutputDir] = useState(() => buildPipelineOutputDir('sft'))
  const [outputDirCustomized, setOutputDirCustomized] = useState(false)
  const [qualityThreshold, setQualityThreshold] = useState('0.7')
  const [numEpochs, setNumEpochs] = useState('3')
  const [useLora, setUseLora] = useState(true)
  const [pushToHub, setPushToHub] = useState('')
  const [deploy, setDeploy] = useState(false)
  const [deployPort, setDeployPort] = useState('8001')
  const [quantization, setQuantization] = useState('4bit')
  const fullPipeline = useRunFullPipeline()

  const [customOpen, setCustomOpen] = useState(false)
  const customPipeline = useRunPipeline()
  const parsedDocPaths = useMemo(() => parseDocumentPaths(multiDocPaths), [multiDocPaths])

  useEffect(() => {
    if (outputDirCustomized) return

    const sourceHint = useMultipleDocuments
      ? parsedDocPaths
      : singleDocPath.trim()
    setOutputDir(buildPipelineOutputDir(technique, sourceHint))
  }, [outputDirCustomized, parsedDocPaths, singleDocPath, technique, useMultipleDocuments])

  function handleRunFull() {
    const docPaths = useMultipleDocuments ? parseDocumentPaths(multiDocPaths) : []
    if (useMultipleDocuments) {
      if (docPaths.length === 0) {
        toast.error('At least one document path is required')
        return
      }
    } else if (!singleDocPath.trim()) {
      toast.error('Document path is required')
      return
    }

    const parsedThreshold = Number(qualityThreshold)
    if (!Number.isFinite(parsedThreshold) || parsedThreshold < 0 || parsedThreshold > 1) {
      toast.error('Quality threshold must be a number between 0 and 1')
      return
    }

    const parsedEpochs = Number(numEpochs)
    if (!Number.isInteger(parsedEpochs) || parsedEpochs < 1) {
      toast.error('Epochs must be a positive integer')
      return
    }

    const parsedDeployPort = Number(deployPort)
    if (deploy && (!Number.isInteger(parsedDeployPort) || parsedDeployPort < 1 || parsedDeployPort > 65535)) {
      toast.error('Deploy port must be an integer between 1 and 65535')
      return
    }

    const args: Record<string, unknown> = {
      technique,
      output_dir: outputDir.trim() || buildPipelineOutputDir(technique, useMultipleDocuments ? docPaths : singleDocPath.trim()),
      quality_threshold: parsedThreshold,
      num_epochs: parsedEpochs,
      use_lora: useLora,
      ...(pushToHub.trim() ? { push_to_hub: pushToHub.trim() } : {}),
      deploy,
      ...(modelPath.trim() ? { base_model: modelPath.trim() } : {}),
      ...(deploy ? { deploy_port: parsedDeployPort } : {}),
      ...(deploy && quantization !== 'none' ? { quantization } : {}),
    }

    if (useMultipleDocuments) {
      if (docPaths.length === 1) args.file_path = docPaths[0]
      else args.file_paths = docPaths
    } else {
      args.file_path = singleDocPath.trim()
    }

    fullPipeline.mutate(args, {
      onSuccess: (data) => {
        setLastResult(data as Record<string, unknown>)
        toast.success('Full pipeline started')
      },
      onError: (err) => toast.error(`Pipeline failed: ${err.message}`),
    })
  }

  function handleRunCustom(args: Record<string, unknown>) {
    customPipeline.mutate(args, {
      onSuccess: (data) => {
        setLastResult(data as Record<string, unknown>)
        toast.success('Custom pipeline started. Track it on Training under Pipeline Training Runs.')
      },
      onError: (err) => toast.error(`Pipeline failed: ${err.message}`),
    })
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-4 w-4" />
            Full Pipeline
          </CardTitle>
          <CardDescription>
            End-to-end: load &rarr; generate &rarr; clean &rarr; normalize &rarr; evaluate &rarr; train
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-md border border-dashed border-border/70 px-3 py-2 text-xs text-muted-foreground">
            Full Pipeline remains document-first and text-only. For `vlm_sft`, use the Custom Pipeline below with a VLM dataset manifest built from the Datasets page.
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground block">Document Source</label>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setUseMultipleDocuments(false)}
                className={cn(
                  'rounded-md border px-3 py-1.5 text-sm transition-colors',
                  !useMultipleDocuments
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-input text-muted-foreground hover:text-foreground',
                )}
              >
                Single Document
              </button>
              <button
                type="button"
                onClick={() => setUseMultipleDocuments(true)}
                className={cn(
                  'rounded-md border px-3 py-1.5 text-sm transition-colors',
                  useMultipleDocuments
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-input text-muted-foreground hover:text-foreground',
                )}
              >
                Multiple Documents
              </button>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-foreground mb-1 block">Document Path</label>
            {useMultipleDocuments ? (
              <DocumentPathListInput
                value={multiDocPaths}
                onChange={setMultiDocPaths}
                disabled={fullPipeline.isPending}
              />
            ) : (
              <DocumentPathInput
                value={singleDocPath}
                onChange={setSingleDocPath}
                placeholder="/path/to/document.pdf"
                disabled={fullPipeline.isPending}
                helperText="Browse uploads a single document to the backend and uses the returned server path for the pipeline."
              />
            )}
          </div>

          <div>
            <label className="text-sm font-medium text-foreground mb-1 block">Technique</label>
            <select
              value={technique}
              onChange={(e) => setTechnique(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {TECHNIQUES.map((t) => (
                <option key={t} value={t}>
                  {t.toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-sm font-medium text-foreground mb-1 block">Model Path</label>
            <ModelPathField
              value={modelPath}
              onChange={setModelPath}
              disabled={fullPipeline.isPending}
              placeholder="meta-llama/Llama-3-8B or ~/.cache/huggingface/hub/..."
              helperText="Use a Hugging Face model ID or browse a backend-visible model folder. Browse defaults to HF Cache when available."
            />
          </div>

          <div>
            <button
              type="button"
              onClick={() => setShowFullAdvanced((open) => !open)}
              className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground cursor-pointer"
            >
              <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', showFullAdvanced && 'rotate-180')} />
              Advanced full_pipeline options
            </button>
            {showFullAdvanced && (
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <div>
                  <label className="text-sm font-medium text-foreground mb-1 block">Output Dir</label>
                  <Input
                    value={outputDir}
                    onChange={(e) => {
                      setOutputDir(e.target.value)
                      setOutputDirCustomized(true)
                    }}
                    placeholder={buildPipelineOutputDir(technique, useMultipleDocuments ? parsedDocPaths : singleDocPath.trim())}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground mb-1 block">Quality Threshold</label>
                  <Input
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={qualityThreshold}
                    onChange={(e) => setQualityThreshold(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground mb-1 block">Epochs</label>
                  <Input
                    type="number"
                    min="1"
                    step="1"
                    value={numEpochs}
                    onChange={(e) => setNumEpochs(e.target.value)}
                  />
                </div>
                <label className="flex items-center gap-2 text-sm text-foreground sm:col-span-2">
                  <input
                    type="checkbox"
                    checked={useLora}
                    onChange={(e) => setUseLora(e.target.checked)}
                    className="h-4 w-4 rounded border-input bg-transparent"
                  />
                  Train with LoRA adapter
                </label>
                <div className="sm:col-span-2">
                  <label className="text-sm font-medium text-foreground mb-1 block">Push To Hub Repo</label>
                  <Input
                    value={pushToHub}
                    onChange={(e) => setPushToHub(e.target.value)}
                    placeholder="your-org/your-model-name"
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    Optional. If set, the trained model or adapter will be pushed to this Hugging Face Hub repo after training.
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground mb-1 block">Deploy Port</label>
                  <Input
                    type="number"
                    min="1"
                    max="65535"
                    step="1"
                    value={deployPort}
                    onChange={(e) => setDeployPort(e.target.value)}
                    disabled={!deploy}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground mb-1 block">Deploy Quantization</label>
                  <select
                    value={quantization}
                    onChange={(e) => setQuantization(e.target.value)}
                    disabled={!deploy}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <option value="4bit">4-bit (recommended, saves memory)</option>
                    <option value="8bit">8-bit</option>
                    <option value="none">None (full precision)</option>
                  </select>
                </div>
                <label className="flex items-center gap-2 text-sm text-foreground sm:col-span-2">
                  <input
                    type="checkbox"
                    checked={deploy}
                    onChange={(e) => setDeploy(e.target.checked)}
                    className="h-4 w-4 rounded border-input bg-transparent"
                  />
                  Deploy after training
                </label>
                {deploy && (
                  <p className="text-xs text-muted-foreground sm:col-span-2">
                    {useLora
                      ? 'Deployment will load the selected base model plus the trained LoRA adapter.'
                      : 'Deployment will load the trained model folder directly because LoRA is disabled.'}{' '}
                    4-bit quantization significantly reduces memory usage. Use &quot;None&quot; only if you have enough
                    VRAM/RAM.
                  </p>
                )}
              </div>
            )}
          </div>

          <Button onClick={handleRunFull} disabled={fullPipeline.isPending}>
            {fullPipeline.isPending ? 'Starting...' : 'Start Full Pipeline'}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <button
            type="button"
            onClick={() => setCustomOpen((v) => !v)}
            className="flex items-center gap-2 w-full text-left cursor-pointer"
          >
            <Settings2 className="h-4 w-4" />
            <CardTitle className="flex-1">Custom Pipeline</CardTitle>
            <ChevronDown
              className={cn('h-4 w-4 transition-transform text-muted-foreground', customOpen && 'rotate-180')}
            />
          </button>
          <CardDescription>Select individual steps and configure each one</CardDescription>
        </CardHeader>
        {customOpen && (
          <CardContent>
            <CustomPipelineForm onSubmit={handleRunCustom} isPending={customPipeline.isPending} />
          </CardContent>
        )}
      </Card>

      {lastResult && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Latest Pipeline Result</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="max-h-80 overflow-auto rounded-md bg-secondary/40 p-3 text-xs font-mono">
              {JSON.stringify(lastResult, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
