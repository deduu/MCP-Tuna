import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { FolderOutput } from 'lucide-react'
import { toast } from 'sonner'
import {
  getDefaultDatasetOutputDir,
  setDefaultDatasetOutputDir,
  resetDefaultDatasetOutputDir,
} from '@/lib/dataset-output'

export function DatasetStorageSection() {
  const [outputDir, setOutputDir] = useState(getDefaultDatasetOutputDir)

  function handleSave() {
    const savedDir = setDefaultDatasetOutputDir(outputDir)
    setOutputDir(savedDir)
    toast.success(`Default dataset output directory set to ${savedDir}`)
  }

  function handleReset() {
    const resetDir = resetDefaultDatasetOutputDir()
    setOutputDir(resetDir)
    toast.success(`Default dataset output directory reset to ${resetDir}`)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FolderOutput className="h-4 w-4" />
          Dataset Storage
        </CardTitle>
        <CardDescription>
          Default save directory used by auto-saved dataset outputs in the UI
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Default output directory</label>
          <div className="flex gap-2">
            <Input
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              placeholder="data"
              onKeyDown={(e) => e.key === 'Enter' && handleSave()}
            />
            <Button onClick={handleSave} disabled={!outputDir.trim()} size="sm">
              Save
            </Button>
            <Button onClick={handleReset} variant="outline" size="sm">
              Reset
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Stored in this browser. It currently applies to Generate from Document,
            clean/normalize outputs, quality-filtered datasets, HF blends, and split/merge defaults.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
