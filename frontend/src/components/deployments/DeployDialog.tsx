import { useState } from 'react'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { useDeploy } from '@/api/hooks/useDeployments'
import { toast } from 'sonner'

interface DeployDialogProps {
  open: boolean
  onClose: () => void
  type: 'mcp' | 'api'
}

export function DeployDialog({ open, onClose, type }: DeployDialogProps) {
  const [modelPath, setModelPath] = useState('')
  const [adapterPath, setAdapterPath] = useState('')
  const [port, setPort] = useState('8001')
  const [quantization, setQuantization] = useState('4bit')

  const deployMutation = useDeploy()

  const handleDeploy = () => {
    if (!modelPath.trim()) {
      toast.error('Model path is required')
      return
    }

    const args: Record<string, unknown> = {
      model_path: modelPath.trim(),
    }

    if (adapterPath.trim()) {
      args.adapter_path = adapterPath.trim()
    }

    args.port = parseInt(port, 10)

    if (quantization !== "none") {
      args.quantization = quantization
    }

    deployMutation.mutate(
      { type, args },
      {
        onSuccess: () => {
          toast.success(`Model deployed as ${type === 'mcp' ? 'MCP server' : 'API endpoint'}`)
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
    setModelPath('')
    setAdapterPath('')
    setPort('8001')
    setQuantization('4bit')
  }

  const title = type === 'mcp' ? 'Deploy as MCP Server' : 'Deploy as API Endpoint'

  return (
    <Dialog open={open} onClose={onClose} title={title}>
      <div className="flex flex-col gap-4">
        {/* Common fields */}
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Model Path <span className="text-destructive">*</span></label>
          <Input
            placeholder="/path/to/model"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Adapter Path</label>
          <Input
            placeholder="/path/to/adapter (optional)"
            value={adapterPath}
            onChange={(e) => setAdapterPath(e.target.value)}
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
            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <option value="4bit">4-bit (recommended, saves memory)</option>
            <option value="8bit">8-bit</option>
            <option value="none">None (full precision)</option>
          </select>
          <p className="text-xs text-muted-foreground">
            4-bit quantization significantly reduces memory usage. Use &quot;None&quot; only if you have enough VRAM/RAM.
          </p>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">Runtime Details</label>
          <p className="rounded-md border border-dashed border-border/70 px-3 py-2 text-sm text-muted-foreground">
            {type === 'mcp'
              ? 'MCP deployments expose a hosted MCP server on the selected port.'
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
