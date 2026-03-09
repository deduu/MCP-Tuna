import { type ChangeEvent, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useTechniques } from '@/api/hooks/useDatasets'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { buildDatasetOutputPath, getDefaultDatasetOutputDir } from '@/lib/dataset-output'
import { FileUp, Sparkles, ChevronDown, ChevronRight, FolderOpen, Loader2 } from 'lucide-react'
import { toast } from 'sonner'

const DOCUMENT_FILE_ACCEPT = '.pdf,.md,.markdown,.txt,.doc,.docx,.json,.jsonl,.csv,.parquet'

interface GenerationSummary {
  count: number
  outputPath: string
}

export function ImportGenerateTab() {
  const queryClient = useQueryClient()
  const { mutateAsync: executeTool, isPending } = useToolExecution()
  const { data: techniques } = useTechniques()

  const [filePath, setFilePath] = useState('')
  const [docPath, setDocPath] = useState('')
  const [technique, setTechnique] = useState('')
  const [loadResult, setLoadResult] = useState<string | null>(null)
  const [genResult, setGenResult] = useState<string | null>(null)
  const [generationSummary, setGenerationSummary] = useState<GenerationSummary | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)

  const [pagePath, setPagePath] = useState('')
  const [batchPath, setBatchPath] = useState('')
  const [schemaResult, setSchemaResult] = useState<string | null>(null)

  const loadFileRef = useRef<HTMLInputElement>(null)
  const generateDocRef = useRef<HTMLInputElement>(null)
  const pageDocRef = useRef<HTMLInputElement>(null)
  const batchDocRef = useRef<HTMLInputElement>(null)

  async function toBase64(file: File): Promise<string> {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        if (typeof reader.result !== 'string') {
          reject(new Error('Failed to read file'))
          return
        }
        const commaIndex = reader.result.indexOf(',')
        resolve(commaIndex >= 0 ? reader.result.slice(commaIndex + 1) : reader.result)
      }
      reader.onerror = () => reject(reader.error ?? new Error('Failed to read file'))
      reader.readAsDataURL(file)
    })
  }

  async function uploadDocument(file: File): Promise<string> {
    const dot = file.name.lastIndexOf('.')
    const baseName = dot >= 0 ? file.name.slice(0, dot) : file.name
    const extension = dot >= 0 ? file.name.slice(dot) : ''
    const safeBaseName = baseName.replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'upload'
    const serverFilename = `documents/${crypto.randomUUID()}_${safeBaseName}${extension}`
    const contentBase64 = await toBase64(file)
    const uploaded = await executeTool({
      toolName: 'file.upload',
      args: { filename: serverFilename, content_base64: contentBase64 },
    })
    const payload = uploaded as Record<string, unknown>
    const uploadedPath = typeof payload.file_path === 'string' ? payload.file_path : ''
    if (!uploadedPath.trim()) {
      throw new Error('Upload succeeded but no server file path was returned')
    }
    return uploadedPath
  }

  async function handlePickSingle(
    event: ChangeEvent<HTMLInputElement>,
    setValue: (path: string) => void,
  ) {
    const file = event.currentTarget.files?.[0]
    if (!file) return

    setIsUploading(true)
    try {
      const uploadedPath = await uploadDocument(file)
      setValue(uploadedPath)
      toast.success(`Uploaded ${file.name}`)
    } catch (err) {
      toast.error(`Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setIsUploading(false)
      event.currentTarget.value = ''
    }
  }

  async function handlePickBatch(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.currentTarget.files ?? [])
    if (!files.length) return

    setIsUploading(true)
    try {
      const uploadedPaths: string[] = []
      for (const file of files) {
        uploadedPaths.push(await uploadDocument(file))
      }
      setBatchPath(uploadedPaths.join('\n'))
      toast.success(`Uploaded ${uploadedPaths.length} document${uploadedPaths.length === 1 ? '' : 's'}`)
    } catch (err) {
      toast.error(`Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setIsUploading(false)
      event.currentTarget.value = ''
    }
  }

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
    setIsGenerating(true)
    setGenerationSummary(null)
    try {
      const result = await executeTool({
        toolName: 'generate.from_document',
        args: { file_path: docPath, technique },
      })

      const payload = result as Record<string, unknown>
      const dataPoints = Array.isArray(payload.data_points)
        ? (payload.data_points as Array<Record<string, unknown>>)
        : []

      if (dataPoints.length === 0) {
        setGenResult(JSON.stringify(payload, null, 2))
        toast.error('Generation completed but produced no rows to save')
        return
      }

      const outputPath = buildDatasetOutputPath(docPath, technique)
      const saved = await executeTool({
        toolName: 'dataset.save',
        args: { data_points: dataPoints, output_path: outputPath, format: 'jsonl' },
      })
      const savedPayload = saved as Record<string, unknown>
      const savedPath =
        typeof savedPayload.file_path === 'string' && savedPayload.file_path.trim()
          ? savedPayload.file_path
          : outputPath

      setGenerationSummary({ count: dataPoints.length, outputPath: savedPath })
      setGenResult(
        JSON.stringify(
          {
            ...payload,
            output_path: savedPath,
            saved: true,
          },
          null,
          2,
        ),
      )
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success(`Dataset generated and saved to ${savedPath}`)
    } catch (err) {
      toast.error(`Generation failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setIsGenerating(false)
    }
  }

  async function handleGeneratePerPage() {
    if (!pagePath.trim() || !technique) return
    try {
      const loaded = await executeTool({
        toolName: 'extract.load_document',
        args: { file_path: pagePath },
      })
      const payload = loaded as Record<string, unknown>
      const pages = Array.isArray(payload.pages) ? payload.pages : []
      const fileName = typeof payload.file_name === 'string' ? payload.file_name : 'document'
      for (let i = 0; i < pages.length; i += 1) {
        await executeTool({
          toolName: 'generate.from_page',
          args: { technique, page_text: String(pages[i]), page_index: i, file_name: fileName },
        })
      }
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success(`Per-page generation complete (${pages.length} pages)`)
    } catch (err) {
      toast.error(`Per-page generation failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleBatchGenerate() {
    if (!batchPath.trim() || !technique) return
    try {
      const filePaths = batchPath
        .split(/[\n,]/)
        .map((s) => s.trim())
        .filter(Boolean)
      await executeTool({
        toolName: 'generate.batch',
        args: { file_paths: filePaths, technique },
      })
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success('Batch generation complete')
    } catch (err) {
      toast.error(`Batch generation failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleViewSchema() {
    if (!technique) {
      toast.error('Select a technique first')
      return
    }
    try {
      const result = await executeTool({
        toolName: 'generate.get_schema',
        args: { technique },
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
            <div className="flex gap-2">
              <Input
                placeholder="File path (e.g., /data/docs/manual.pdf)"
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
              />
              <Button
                type="button"
                variant="outline"
                onClick={() => loadFileRef.current?.click()}
                disabled={isUploading}
              >
                <FolderOpen className="h-4 w-4" />
                {isUploading ? 'Uploading...' : 'Browse'}
              </Button>
              <input
                ref={loadFileRef}
                type="file"
                className="hidden"
                accept={DOCUMENT_FILE_ACCEPT}
                onChange={(e) => handlePickSingle(e, setFilePath)}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Browse uploads the file to the backend and fills in a server path automatically.
            </p>
            <Button onClick={handleLoadDocument} disabled={isPending || isUploading || !filePath.trim()}>
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
            <div className="flex gap-2">
              <Input
                placeholder="Document path"
                value={docPath}
                onChange={(e) => setDocPath(e.target.value)}
              />
              <Button
                type="button"
                variant="outline"
                onClick={() => generateDocRef.current?.click()}
                disabled={isUploading}
              >
                <FolderOpen className="h-4 w-4" />
                {isUploading ? 'Uploading...' : 'Browse'}
              </Button>
              <input
                ref={generateDocRef}
                type="file"
                className="hidden"
                accept={DOCUMENT_FILE_ACCEPT}
                onChange={(e) => handlePickSingle(e, setDocPath)}
              />
            </div>
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
            <Button
              onClick={handleGenerate}
              disabled={isPending || isUploading || !docPath.trim() || !technique}
            >
              {isGenerating && <Loader2 className="h-4 w-4 animate-spin" />}
              {isGenerating ? 'Generating...' : 'Generate'}
            </Button>
            {isGenerating && (
              <p className="text-xs text-muted-foreground">
                Generating dataset and saving it to {getDefaultDatasetOutputDir()}...
              </p>
            )}
            {generationSummary && (
              <div className="space-y-2 rounded-md border border-border/60 bg-secondary/20 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="success">Generated: {generationSummary.count} rows</Badge>
                  <Badge variant="outline">
                    Saved: {generationSummary.outputPath.split(/[\\/]/).pop()}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground break-all">
                  {generationSummary.outputPath}
                </p>
              </div>
            )}
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
                <div className="flex gap-2">
                  <Input
                    placeholder="Document path"
                    value={pagePath}
                    onChange={(e) => setPagePath(e.target.value)}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => pageDocRef.current?.click()}
                    disabled={isUploading}
                  >
                    <FolderOpen className="h-4 w-4" />
                    {isUploading ? 'Uploading...' : 'Browse'}
                  </Button>
                  <input
                    ref={pageDocRef}
                    type="file"
                    className="hidden"
                    accept={DOCUMENT_FILE_ACCEPT}
                    onChange={(e) => handlePickSingle(e, setPagePath)}
                  />
                </div>
                <Button
                  onClick={handleGeneratePerPage}
                  disabled={isPending || isUploading || !pagePath.trim() || !technique}
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
                <div className="flex gap-2">
                  <Input
                    placeholder="/path/doc1.pdf, /path/doc2.md"
                    value={batchPath}
                    onChange={(e) => setBatchPath(e.target.value)}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => batchDocRef.current?.click()}
                    disabled={isUploading}
                  >
                    <FolderOpen className="h-4 w-4" />
                    {isUploading ? 'Uploading...' : 'Browse'}
                  </Button>
                  <input
                    ref={batchDocRef}
                    type="file"
                    className="hidden"
                    accept={DOCUMENT_FILE_ACCEPT}
                    multiple
                    onChange={handlePickBatch}
                  />
                </div>
                <Button
                  onClick={handleBatchGenerate}
                  disabled={isPending || isUploading || !batchPath.trim() || !technique}
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
