import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useTechniques } from '@/api/hooks/useDatasets'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { FileUp, Sparkles, ChevronDown, ChevronRight } from 'lucide-react'
import { toast } from 'sonner'

export function ImportGenerateTab() {
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const { data: techniques } = useTechniques()

  const [filePath, setFilePath] = useState('')
  const [docPath, setDocPath] = useState('')
  const [technique, setTechnique] = useState('')
  const [numSamples, setNumSamples] = useState(10)
  const [loadResult, setLoadResult] = useState<string | null>(null)
  const [genResult, setGenResult] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Advanced fields
  const [pagePath, setPagePath] = useState('')
  const [batchPath, setBatchPath] = useState('')
  const [schemaResult, setSchemaResult] = useState<string | null>(null)

  async function handleLoadDocument() {
    if (!filePath.trim()) return
    try {
      const result = await executeTool({
        toolName: 'extract.load_document',
        args: { file_path: filePath },
      })
      setLoadResult(JSON.stringify(result, null, 2))
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success('Document loaded successfully')
    } catch (err) {
      toast.error(`Load failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleGenerate() {
    if (!docPath.trim() || !technique) return
    try {
      const result = await executeTool({
        toolName: 'generate.from_document',
        args: { document_path: docPath, technique, num_samples: numSamples },
      })
      setGenResult(JSON.stringify(result, null, 2))
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success('Dataset generated successfully')
    } catch (err) {
      toast.error(`Generation failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleGeneratePerPage() {
    if (!pagePath.trim() || !technique) return
    try {
      await executeTool({
        toolName: 'generate.from_page',
        args: { document_path: pagePath, technique, num_samples: numSamples },
      })
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success('Per-page generation complete')
    } catch (err) {
      toast.error(`Per-page generation failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleBatchGenerate() {
    if (!batchPath.trim() || !technique) return
    try {
      await executeTool({
        toolName: 'generate.batch',
        args: { document_path: batchPath, technique, num_samples: numSamples },
      })
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success('Batch generation complete')
    } catch (err) {
      toast.error(`Batch generation failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleViewSchema() {
    try {
      const result = await executeTool({
        toolName: 'generate.get_schema',
        args: {},
      })
      setSchemaResult(JSON.stringify(result, null, 2))
    } catch (err) {
      toast.error(`Failed to load schema: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Document */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <FileUp className="h-4 w-4" />
              Upload Document
            </CardTitle>
            <CardDescription>Load a document to create a dataset</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              placeholder="File path (e.g., /data/docs/manual.pdf)"
              value={filePath}
              onChange={(e) => setFilePath(e.target.value)}
            />
            <Button onClick={handleLoadDocument} disabled={isPending || !filePath.trim()}>
              Load Document
            </Button>
            {loadResult && (
              <pre className="text-xs bg-secondary/50 rounded p-3 overflow-auto max-h-40">
                {loadResult}
              </pre>
            )}
          </CardContent>
        </Card>

        {/* Generate from Document */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Sparkles className="h-4 w-4" />
              Generate from Document
            </CardTitle>
            <CardDescription>Generate training data from a loaded document</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              placeholder="Document path"
              value={docPath}
              onChange={(e) => setDocPath(e.target.value)}
            />
            <select
              value={technique}
              onChange={(e) => setTechnique(e.target.value)}
              className="w-full h-9 rounded-md border border-input bg-transparent px-3 text-sm text-foreground"
            >
              <option value="">Select technique...</option>
              {techniques?.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <Input
              type="number"
              placeholder="Number of samples"
              value={numSamples}
              onChange={(e) => setNumSamples(Number(e.target.value))}
              min={1}
            />
            <Button
              onClick={handleGenerate}
              disabled={isPending || !docPath.trim() || !technique}
            >
              Generate
            </Button>
            {genResult && (
              <pre className="text-xs bg-secondary/50 rounded p-3 overflow-auto max-h-40">
                {genResult}
              </pre>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Advanced Section */}
      <div>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
        >
          {showAdvanced ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          Show Advanced
        </button>

        {showAdvanced && (
          <div className="mt-4 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Generate Per Page</CardTitle>
                <CardDescription>Generate data from each page independently</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Input
                  placeholder="Document path"
                  value={pagePath}
                  onChange={(e) => setPagePath(e.target.value)}
                />
                <Button
                  onClick={handleGeneratePerPage}
                  disabled={isPending || !pagePath.trim() || !technique}
                  variant="secondary"
                >
                  Generate Per Page
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Batch Generate</CardTitle>
                <CardDescription>Generate from multiple documents at once</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Input
                  placeholder="Directory or glob path"
                  value={batchPath}
                  onChange={(e) => setBatchPath(e.target.value)}
                />
                <Button
                  onClick={handleBatchGenerate}
                  disabled={isPending || !batchPath.trim() || !technique}
                  variant="secondary"
                >
                  Batch Generate
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">View Schema</CardTitle>
                <CardDescription>View the generation output schema</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button onClick={handleViewSchema} disabled={isPending} variant="secondary">
                  View Schema
                </Button>
                {schemaResult && (
                  <pre className="text-xs bg-secondary/50 rounded p-3 overflow-auto max-h-60">
                    {schemaResult}
                  </pre>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}
