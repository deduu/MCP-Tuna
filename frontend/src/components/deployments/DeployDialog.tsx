import { useEffect, useState } from 'react'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { useDeploy } from '@/api/hooks/useDeployments'
import { toast } from 'sonner'
import { ModelPathField } from '@/components/pipeline/ModelPathField'

export interface DeployDialogInitialValues {
  name?: string
  systemPrompt?: string | null
  modelPath?: string
  adapterPath?: string
  port?: number
  quantization?: '4bit' | '8bit' | 'none'
  modality?: 'text' | 'vision-language'
}

interface DeployDialogProps {
  open: boolean
  onClose: () => void
  type: 'mcp' | 'api'
  initialValues?: DeployDialogInitialValues | null
}

function basename(value: string): string {
  const normalized = value.trim().replace(/\\/g, '/')
  const leaf = normalized.split('/').pop() || normalized
  return leaf.replace(/\.[^.]+$/, '')
}

function inferDeploymentName(modelPath: string, adapterPath: string): string {
  const modelName = basename(modelPath)
  const adapterName = basename(adapterPath)

  if (adapterName) {
    if (!modelName || adapterName === modelName) return adapterName
    return `${modelName} + ${adapterName}`
  }

  return modelName
}

export function DeployDialog({ open, onClose, type, initialValues }: DeployDialogProps) {
  const [name, setName] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [adapterPath, setAdapterPath] = useState('')
  const [port, setPort] = useState('8001')
  const [quantization, setQuantization] = useState('4bit')
  const [modality, setModality] = useState<'text' | 'vision-language'>('text')
  const [nameCustomized, setNameCustomized] = useState(false)

  const deployMutation = useDeploy()

  useEffect(() => {
    if (!open) return

    const nextModelPath = initialValues?.modelPath ?? ''
    const nextAdapterPath = initialValues?.adapterPath ?? ''
    const nextName = initialValues?.name ?? inferDeploymentName(nextModelPath, nextAdapterPath)

    setName(nextName)
    setSystemPrompt(initialValues?.systemPrompt ?? '')
    setModelPath(initialValues?.modelPath ?? '')
    setAdapterPath(initialValues?.adapterPath ?? '')
    setPort(String(initialValues?.port ?? 8001))
    setQuantization(initialValues?.quantization ?? '4bit')
    setModality(initialValues?.modality ?? 'text')
    setNameCustomized(Boolean(initialValues?.name))
  }, [open, initialValues, type])

  useEffect(() => {
    if (nameCustomized) return
    setName(inferDeploymentName(modelPath, adapterPath))
  }, [modelPath, adapterPath, nameCustomized])

  const handleDeploy = () => {
    if (!modelPath.trim()) {
      toast.error('Model path is required')
      return
    }

    const args: Record<string, unknown> = {
      model_path: modelPath.trim(),
    }

    if (name.trim()) {
      args.name = name.trim()
    }
    if (systemPrompt.trim()) {
      args.system_prompt = systemPrompt.trim()
    }

    if (adapterPath.trim()) {
      args.adapter_path = adapterPath.trim()
    }

    args.port = parseInt(port, 10)

    if (modality === 'text' && quantization !== "none") {
      args.quantization = quantization
    }

    deployMutation.mutate(
      { type, modality, args },
      {
        onSuccess: () => {
          toast.success(`${modality === 'vision-language' ? 'Vision-language' : 'Text'} model deployed as ${type === 'mcp' ? 'MCP server' : 'API endpoint'}`)
          resetForm()
          onClose()
        },
        onError: (err) => {
          toast.error(`Deployment failed: ${err.message}`)
        },
      },
    )
  }

  const resetForm = () => {
    setName('')
    setSystemPrompt('')
    setModelPath('')
    setAdapterPath('')
    setPort('8001')
    setQuantization('4bit')
    setModality('text')
    setNameCustomized(false)
  }

  const title = type === 'mcp' ? 'Deploy as MCP Server' : 'Deploy as API Endpoint'

  return (
    <Dialog open={open} onClose={onClose} title={title} className="max-w-2xl">
      <div className="flex flex-col gap-4">
        <div className="rounded-md border border-dashed border-border/70 px-3 py-2 text-sm text-muted-foreground">
          Base-model-only deployment is supported. For LoRA outputs, set the original base model in Model Path and the trained adapter folder in Adapter Path.
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Deployment Name</label>
          <Input
            value={name}
            onChange={(e) => {
              setName(e.target.value)
              setNameCustomized(true)
            }}
            placeholder="Optional display name"
          />
          <p className="text-xs text-muted-foreground">
            Optional. Defaults to the model name, or a base-model plus adapter label for LoRA deployments.
          </p>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Default System Prompt</label>
          <textarea
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={4}
            placeholder="Optional. Set a default instruction for deployment chat."
            className="flex min-h-[96px] w-full resize-y rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
          <p className="text-xs text-muted-foreground">
            Optional. This is reused as the default chat system prompt for this deployment and can still be overridden per conversation.
          </p>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Modality</label>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setModality('text')}
              className={`rounded-md border px-3 py-1.5 text-sm transition-colors ${
                modality === 'text'
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-input text-muted-foreground hover:text-foreground'
              }`}
            >
              Text
            </button>
            <button
              type="button"
              onClick={() => setModality('vision-language')}
              className={`rounded-md border px-3 py-1.5 text-sm transition-colors ${
                modality === 'vision-language'
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-input text-muted-foreground hover:text-foreground'
              }`}
            >
              Vision-Language
            </button>
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Model Path <span className="text-destructive">*</span></label>
          <ModelPathField
            value={modelPath}
            onChange={setModelPath}
            placeholder={
              modality === 'vision-language'
                ? 'Qwen/Qwen2.5-VL-3B-Instruct or C:/models/vlm-base'
                : 'meta-llama/Llama-3.2-1B-Instruct or C:/models/base'
            }
            helperText={
              modality === 'vision-language'
                ? 'Vision-language base model folder or Hugging Face model ID.'
                : 'Base model folder or Hugging Face model ID.'
            }
            validationPurpose="model"
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Adapter Path</label>
          <ModelPathField
            value={adapterPath}
            onChange={setAdapterPath}
            placeholder="C:/output/my-lora-adapter"
            helperText="Optional. Leave empty to deploy the base model directly."
            validationPurpose="adapter"
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Port</label>
          <Input
            type="number"
            placeholder="8001"
            value={port}
            onChange={(e) => setPort(e.target.value)}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Quantization</label>
          <select
            value={quantization}
            onChange={(e) => setQuantization(e.target.value)}
            disabled={modality === 'vision-language'}
            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <option value="4bit">4-bit (recommended, saves memory)</option>
            <option value="8bit">8-bit</option>
            <option value="none">None (full precision)</option>
          </select>
          <p className="text-xs text-muted-foreground">
            {modality === 'vision-language'
              ? 'Current VLM deployment uses the multimodal inference path and ignores quantization overrides for now.'
              : '4-bit quantization significantly reduces memory usage. Use "None" only if you have enough VRAM/RAM.'}
          </p>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Runtime Details</label>
          <p className="rounded-md border border-dashed border-border/70 px-3 py-2 text-sm text-muted-foreground">
            {type === 'mcp'
              ? modality === 'vision-language'
                ? 'VLM MCP deployments expose a hosted MCP server with a generate_vlm tool.'
                : 'MCP deployments expose a hosted MCP server on the selected port.'
              : modality === 'vision-language'
                ? 'VLM API deployments expose /generate_vlm and /health.'
                : 'API deployments expose fixed routes: /generate and /health.'}
          </p>
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2 pt-2">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleDeploy} disabled={deployMutation.isPending}>
            {deployMutation.isPending ? 'Deploying...' : 'Deploy'}
          </Button>
        </div>
      </div>
    </Dialog>
  )
}
