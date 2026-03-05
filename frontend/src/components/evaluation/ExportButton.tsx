import { useState } from 'react'
import { Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { mcpCall } from '@/api/client'
import { toast } from 'sonner'

interface ExportButtonProps {
  toolName: string
  args: Record<string, unknown>
  label?: string
}

export function ExportButton({ toolName, args, label = 'Export' }: ExportButtonProps) {
  const [loading, setLoading] = useState(false)

  async function handleExport() {
    setLoading(true)
    try {
      await mcpCall(toolName, args)
      toast.success('Results exported')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Button variant="outline" size="sm" onClick={handleExport} disabled={loading}>
      <Download className="h-3.5 w-3.5" />
      {loading ? 'Exporting...' : label}
    </Button>
  )
}
