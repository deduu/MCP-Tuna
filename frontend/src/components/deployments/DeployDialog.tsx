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
  // MCP fields
  const [toolName, setToolName] = useState('')
  const [description, setDescription] = useState('')
  // API fields
  const [port, setPort] = useState('8080')
  const [route, setRoute] = useState('/v1/chat/completions')

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

    if (type === 'mcp') {
      if (toolName.trim()) args.tool_name = toolName.trim()
      if (description.trim()) args.description = description.trim()
    } else {
      args.port = parseInt(port, 10)
      if (route.trim()) args.route = route.trim()
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
    setToolName('')
    setDescription('')
    setPort('8080')
    setRoute('/v1/chat/completions')
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

        {/* MCP-specific fields */}
        {type === 'mcp' && (
          <>
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">Tool Name</label>
              <Input
                placeholder="my_model_tool"
                value={toolName}
                onChange={(e) => setToolName(e.target.value)}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">Description</label>
              <textarea
                className="flex min-h-[80px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring resize-none"
                placeholder="Describe what this MCP tool does..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>
          </>
        )}

        {/* API-specific fields */}
        {type === 'api' && (
          <>
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">Port</label>
              <Input
                type="number"
                placeholder="8080"
                value={port}
                onChange={(e) => setPort(e.target.value)}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">Route</label>
              <Input
                placeholder="/v1/chat/completions"
                value={route}
                onChange={(e) => setRoute(e.target.value)}
              />
            </div>
          </>
        )}

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
