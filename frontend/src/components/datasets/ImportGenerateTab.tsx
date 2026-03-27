import { type ChangeEvent, useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { mcpCall } from '@/api/client'
import { useTechniques } from '@/api/hooks/useDatasets'
import { useToolExecution } from '@/api/hooks/useToolExecution'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { buildDatasetOutputPath, getDefaultDatasetOutputDir } from '@/lib/dataset-output'
import {
  FileUp,
  Sparkles,
  ChevronDown,
  ChevronRight,
  FolderOpen,
  Loader2,
  Plus,
  Trash2,
} from 'lucide-react'
import { toast } from 'sonner'
import { VlmDatasetBuilder } from './VlmDatasetBuilder'

const DOCUMENT_FILE_ACCEPT = '.pdf,.md,.markdown,.txt,.doc,.docx,.json,.jsonl,.csv,.parquet'
const HF_TARGET_FORMAT_OPTIONS = ['raw', 'sft', 'dpo'] as const

interface HfCustomSource {
  id: string
  datasetName: string
  subset: string
  split: string
  maxRows: string
  renameColumns: string
  dropColumns: string
}

interface GenerationSummary {
  count: number
  outputPath: string
}

interface HfBlendJobPayload {
  success?: boolean
  job_id?: string
  status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  error?: string
  progress?: {
    percent_complete?: number
    status_message?: string
  }
  result?: {
    count?: number
    save_result?: {
      file_path?: string
    }
  }
}

function createEmptyHfSource(): HfCustomSource {
  return {
    id: crypto.randomUUID(),
    datasetName: '',
    subset: '',
    split: 'train',
    maxRows: '',
    renameColumns: '',
    dropColumns: '',
  }
}

function buildDefaultHfOutputPath(recipeName: string | null, targetFormat: string): string {
  const outputDir = getDefaultDatasetOutputDir()
  if (recipeName) {
    return `${outputDir}/${recipeName}.jsonl`
  }
  return `${outputDir}/hf_blend_${targetFormat}.jsonl`
}

function validateCustomHfSource(source: HfCustomSource, index: number): void {
  const datasetName = source.datasetName.trim()
  const subset = source.subset.trim()
  const split = source.split.trim()
  const maxRows = source.maxRows.trim()

  if (!datasetName) {
    throw new Error(`Source ${index + 1} is missing a dataset name`)
  }

  if (datasetName.includes(' / ')) {
    throw new Error(
      `Source ${index + 1} dataset name must be only the Hub id, for example ` +
        `'HuggingFaceTB/smoltalk2'. Put subset and split in their own fields.`,
    )
  }

  if (/^\d+$/.test(subset) && !maxRows) {
    throw new Error(
      `Source ${index + 1} subset looks like a row cap. Move '${subset}' to Max Rows.`,
    )
  }

  if (datasetName === 'HuggingFaceTB/smoltalk2' && !subset) {
    throw new Error(
      `Source ${index + 1} needs subset 'SFT' for current smoltalk2 slices. Put the slice name in Split.`,
    )
  }

  if (datasetName === 'HuggingFaceTB/smoltalk2' && split.toLowerCase() === 'train') {
    throw new Error(
      `Source ${index + 1} split cannot be 'train' for the smoltalk2 slice view. Use a named split like ` +
        `'multi_turn_reasoning_if_think' or 'smoltalk_smollm3_smol_magpie_ultra_no_think'.`,
    )
  }
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
  const [isLoadingTemplate, setIsLoadingTemplate] = useState(false)
  const [customTemplate, setCustomTemplate] = useState('')
  const [startPage, setStartPage] = useState('')
  const [endPage, setEndPage] = useState('')

  const [pagePath, setPagePath] = useState('')
  const [batchPath, setBatchPath] = useState('')
  const [schemaResult, setSchemaResult] = useState<string | null>(null)
  const [hfTargetFormat, setHfTargetFormat] = useState<(typeof HF_TARGET_FORMAT_OPTIONS)[number]>('sft')
  const [hfCustomSources, setHfCustomSources] = useState<HfCustomSource[]>([createEmptyHfSource()])
  const [hfMaxRowsPerSource, setHfMaxRowsPerSource] = useState('')
  const [hfOutputPath, setHfOutputPath] = useState(buildDefaultHfOutputPath(null, 'sft'))
  const [hfComposeResult, setHfComposeResult] = useState<string | null>(null)
  const [hfGenerationSummary, setHfGenerationSummary] = useState<GenerationSummary | null>(null)
  const [isComposingRecipe, setIsComposingRecipe] = useState(false)
  const [hfBlendJobId, setHfBlendJobId] = useState<string | null>(null)
  const [hfBlendJobStatus, setHfBlendJobStatus] = useState<HfBlendJobPayload | null>(null)

  const loadFileRef = useRef<HTMLInputElement>(null)
  const generateDocRef = useRef<HTMLInputElement>(null)
  const pageDocRef = useRef<HTMLInputElement>(null)
  const batchDocRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!hfBlendJobId) {
      return
    }

    let cancelled = false
    let timeoutId: number | undefined

    async function pollJob() {
      try {
        const job = await mcpCall<HfBlendJobPayload>('generate.hf_blend_job_status', { job_id: hfBlendJobId })
        if (cancelled) {
          return
        }

        setHfBlendJobStatus(job)
        setHfComposeResult(JSON.stringify(job, null, 2))

        if (job.status === 'completed') {
          const savedPath = job.result?.save_result?.file_path ?? hfOutputPath
          const count = typeof job.result?.count === 'number' ? job.result.count : 0
          setHfOutputPath(savedPath)
          setHfGenerationSummary({ count, outputPath: savedPath })
          setIsComposingRecipe(false)
          setHfBlendJobId(null)
          queryClient.invalidateQueries({ queryKey: ['datasets'] })
          toast.success(`HF dataset blend saved to ${savedPath}`)
          return
        }

        if (job.status === 'failed' || job.status === 'cancelled') {
          setIsComposingRecipe(false)
          setHfBlendJobId(null)
          toast.error(job.error || `HF dataset blend ${job.status}`)
          return
        }

        timeoutId = window.setTimeout(() => {
          void pollJob()
        }, 2_000)
      } catch (err) {
        if (cancelled) {
          return
        }
        setIsComposingRecipe(false)
        setHfBlendJobId(null)
        toast.error(`Failed to monitor HF dataset blend: ${err instanceof Error ? err.message : 'Unknown error'}`)
      }
    }

    void pollJob()

    return () => {
      cancelled = true
      if (timeoutId !== undefined) {
        window.clearTimeout(timeoutId)
      }
    }
  }, [hfBlendJobId, hfOutputPath, queryClient])

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

  function parseOptionalPage(value: string): number | undefined {
    const trimmed = value.trim()
    if (!trimmed) return undefined
    const parsed = Number(trimmed)
    return Number.isInteger(parsed) && parsed >= 0 ? parsed : undefined
  }

  function validatePromptControls(): boolean {
    if (customTemplate.trim() && !customTemplate.includes('{text}')) {
      toast.error('Custom template must include the {text} placeholder')
      return false
    }

    const parsedStart = parseOptionalPage(startPage)
    const parsedEnd = parseOptionalPage(endPage)

    if (startPage.trim() && parsedStart === undefined) {
      toast.error('Start page must be a non-negative integer')
      return false
    }
    if (endPage.trim() && parsedEnd === undefined) {
      toast.error('End page must be a non-negative integer')
      return false
    }
    if (parsedStart !== undefined && parsedEnd !== undefined && parsedEnd < parsedStart) {
      toast.error('End page must be greater than or equal to start page')
      return false
    }

    return true
  }

  function buildGenerateArgs(baseArgs: Record<string, unknown>) {
    const args: Record<string, unknown> = { ...baseArgs }
    const parsedStart = parseOptionalPage(startPage)
    const parsedEnd = parseOptionalPage(endPage)

    if (customTemplate.trim()) {
      args.custom_template = customTemplate
    }
    if (parsedStart !== undefined) {
      args.start_page = parsedStart
    }
    if (parsedEnd !== undefined) {
      args.end_page = parsedEnd
    }

    return args
  }

  function buildTemplateArgs(baseArgs: Record<string, unknown>) {
    const args: Record<string, unknown> = { ...baseArgs }
    if (customTemplate.trim()) {
      args.custom_template = customTemplate
    }
    return args
  }

  async function handleLoadDefaultTemplate() {
    if (!technique) {
      toast.error('Select a technique first')
      return
    }

    setIsLoadingTemplate(true)
    try {
      const result = await executeTool({
        toolName: 'generate.get_template',
        args: { technique },
      })
      const payload = result as Record<string, unknown>
      const template = typeof payload.template === 'string' ? payload.template : ''
      if (!template) {
        throw new Error(typeof payload.error === 'string' ? payload.error : 'Template not found')
      }
      setCustomTemplate(template)
      toast.success(`Loaded default ${technique} template`)
    } catch (err) {
      toast.error(`Failed to load template: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setIsLoadingTemplate(false)
    }
  }

  async function handleGenerate() {
    if (!docPath.trim() || !technique) return
    if (!validatePromptControls()) return
    setIsGenerating(true)
    setGenerationSummary(null)
    try {
      const result = await executeTool({
        toolName: 'generate.from_document',
        args: buildGenerateArgs({ file_path: docPath, technique }),
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
    if (!validatePromptControls()) return
    try {
      const loaded = await executeTool({
        toolName: 'extract.load_document',
        args: { file_path: pagePath },
      })
      const payload = loaded as Record<string, unknown>
      const pages = Array.isArray(payload.pages) ? payload.pages : []
      const fileName = typeof payload.file_name === 'string' ? payload.file_name : 'document'
      const parsedStart = parseOptionalPage(startPage) ?? 0
      const parsedEnd = parseOptionalPage(endPage) ?? pages.length - 1

      for (let i = parsedStart; i < pages.length && i <= parsedEnd; i += 1) {
        const page = pages[i] as Record<string, unknown>
        const pageText =
          typeof page?.markdown === 'string'
            ? page.markdown
            : typeof page?.text === 'string'
              ? page.text
              : String(page)
        await executeTool({
          toolName: 'generate.from_page',
          args: buildTemplateArgs({
            technique,
            page_text: pageText,
            page_index: i,
            file_name: fileName,
          }),
        })
      }
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      toast.success('Per-page generation complete')
    } catch (err) {
      toast.error(`Per-page generation failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleBatchGenerate() {
    if (!batchPath.trim() || !technique) return
    if (!validatePromptControls()) return
    try {
      const filePaths = batchPath
        .split(/[\n,]/)
        .map((s) => s.trim())
        .filter(Boolean)
      await executeTool({
        toolName: 'generate.batch',
        args: buildTemplateArgs({ file_paths: filePaths, technique }),
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

  async function handleComposeRecipe() {
    setIsComposingRecipe(true)
    setHfGenerationSummary(null)
    setHfBlendJobStatus(null)
    try {
      const parsedCap = Number.parseInt(hfMaxRowsPerSource, 10)
      const maxRowsPerSource =
        Number.isFinite(parsedCap) && parsedCap > 0 ? parsedCap : undefined
      const defaultOutputPath = buildDefaultHfOutputPath(null, hfTargetFormat)
      const outputPath = hfOutputPath.trim() || defaultOutputPath

      const sources = hfCustomSources.map((source, index) => {
        validateCustomHfSource(source, index)

        const datasetName = source.datasetName.trim()

        let renameColumns: Record<string, string> | undefined
        if (source.renameColumns.trim()) {
          const parsed = JSON.parse(source.renameColumns)
          if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            throw new Error(`Source ${index + 1} rename columns must be a JSON object`)
          }
          renameColumns = parsed as Record<string, string>
        }

        return {
          dataset_name: datasetName,
          subset: source.subset.trim() || undefined,
          split: source.split.trim() || 'train',
          max_rows:
            Number.isFinite(Number.parseInt(source.maxRows, 10)) && Number.parseInt(source.maxRows, 10) > 0
              ? Number.parseInt(source.maxRows, 10)
              : undefined,
          rename_columns: renameColumns,
          drop_columns: source.dropColumns
            .split(',')
            .map((value) => value.trim())
            .filter(Boolean),
        }
      })

      const args: Record<string, unknown> = {
        output_path: outputPath,
        format: 'jsonl',
        sources: JSON.stringify(sources),
        target_format: hfTargetFormat,
        shuffle: true,
        seed: 42,
        max_rows_per_source: maxRowsPerSource,
      }

      const result = await executeTool({
        toolName: 'generate.compose_hf_dataset_async',
        args,
      })

      const payload = result as HfBlendJobPayload
      if (!payload.job_id) {
        setHfComposeResult(JSON.stringify(payload, null, 2))
        setIsComposingRecipe(false)
        toast.error('HF dataset blend did not return a job id')
        return
      }
      setHfComposeResult(JSON.stringify(payload, null, 2))
      setHfBlendJobId(payload.job_id)
      toast.success('HF dataset blend started in the background')
    } catch (err) {
      toast.error(`HF dataset blend failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
      setIsComposingRecipe(false)
    } finally {
      // Poller owns the running state after job submission succeeds.
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
            <div className="space-y-3 rounded-md border border-border/60 bg-secondary/20 p-3">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">Start Page / Chunk</label>
                  <Input
                    type="number"
                    min="0"
                    placeholder="0"
                    value={startPage}
                    onChange={(e) => setStartPage(e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">End Page / Chunk</label>
                  <Input
                    type="number"
                    min="0"
                    placeholder="Last"
                    value={endPage}
                    onChange={(e) => setEndPage(e.target.value)}
                  />
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleLoadDefaultTemplate}
                  disabled={isPending || isLoadingTemplate || !technique}
                >
                  {isLoadingTemplate ? 'Loading template...' : 'Load Default Template'}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setCustomTemplate('')}
                  disabled={!customTemplate}
                >
                  Clear Custom Template
                </Button>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">Custom Prompt Template</label>
                <textarea
                  value={customTemplate}
                  onChange={(e) => setCustomTemplate(e.target.value)}
                  placeholder={'Use the default template or write your own. Keep the {text} placeholder so the document chunk is injected.'}
                  className="min-h-48 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40"
                />
                <p className="text-xs text-muted-foreground">
                  Leave this blank to use the built-in template. For finer control, load the default
                  template, then edit wording, density requirements, or answer style.
                </p>
              </div>
            </div>
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

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Sparkles className="h-4 w-4" />
            HF Dataset Blend
          </CardTitle>
          <CardDescription>
            Compose one or more Hugging Face datasets into training-ready rows.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Target Format</label>
              <select
                value={hfTargetFormat}
                onChange={(e) => {
                  const next = e.target.value as (typeof HF_TARGET_FORMAT_OPTIONS)[number]
                  setHfTargetFormat(next)
                  setHfOutputPath(buildDefaultHfOutputPath(null, next))
                }}
                className="w-full h-9 rounded-md border border-input bg-transparent px-3 text-sm text-foreground"
              >
                {HF_TARGET_FORMAT_OPTIONS.map((format) => (
                  <option key={format} value={format}>
                    {format.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Global Row Cap</label>
              <Input
                type="number"
                min="1"
                value={hfMaxRowsPerSource}
                onChange={(e) => setHfMaxRowsPerSource(e.target.value)}
                placeholder="Optional per-source limit"
              />
            </div>
          </div>
          <div className="rounded-md border border-border/60 bg-secondary/20 p-3 space-y-2">
            <p className="text-xs text-muted-foreground">
              Dataset Name is the Hub repo id only. Subset is the config name when the dataset has
              one. Split can be a standard split like `train` or a named slice.
            </p>
            <p className="text-xs text-muted-foreground">
              Example: Dataset Name `HuggingFaceTB/smoltalk2`, Subset `SFT`, Split
              `multi_turn_reasoning_if_think`, Max Rows `100`, Drop Columns `chat_template_kwargs`.
            </p>
            <p className="text-xs text-muted-foreground">
              Example stage datasets if you want published TRLM sources directly:
              `Shekswess/trlm-sft-stage-1-final-2`, `Shekswess/trlm-sft-stage-2-final-2`,
              `Shekswess/trlm-dpo-stage-3-final-2`.
            </p>
          </div>
          <div className="space-y-3">
            {hfCustomSources.map((source, index) => (
              <div key={source.id} className="space-y-3 rounded-md border border-border/60 p-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="text-sm font-medium">Source {index + 1}</div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setHfCustomSources((current) =>
                        current.length === 1 ? current : current.filter((row) => row.id !== source.id),
                      )
                    }}
                    disabled={hfCustomSources.length === 1}
                  >
                    <Trash2 className="h-4 w-4" />
                    Remove
                  </Button>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-1 md:col-span-2">
                    <label className="text-xs font-medium text-muted-foreground">Dataset Name</label>
                    <Input
                      value={source.datasetName}
                      onChange={(e) =>
                        setHfCustomSources((current) =>
                          current.map((row) =>
                            row.id === source.id ? { ...row, datasetName: e.target.value } : row,
                          ),
                        )
                      }
                      placeholder="HuggingFaceTB/smoltalk2"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-medium text-muted-foreground">Subset / Config</label>
                    <Input
                      value={source.subset}
                      onChange={(e) =>
                        setHfCustomSources((current) =>
                          current.map((row) =>
                            row.id === source.id ? { ...row, subset: e.target.value } : row,
                          ),
                        )
                      }
                      placeholder="optional subset, e.g. SFT"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-medium text-muted-foreground">Split / Slice</label>
                    <Input
                      value={source.split}
                      onChange={(e) =>
                        setHfCustomSources((current) =>
                          current.map((row) =>
                            row.id === source.id ? { ...row, split: e.target.value } : row,
                          ),
                        )
                      }
                      placeholder="train or named slice"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-medium text-muted-foreground">Max Rows</label>
                    <Input
                      type="number"
                      min="1"
                      value={source.maxRows}
                      onChange={(e) =>
                        setHfCustomSources((current) =>
                          current.map((row) =>
                            row.id === source.id ? { ...row, maxRows: e.target.value } : row,
                          ),
                        )
                      }
                      placeholder="optional"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-medium text-muted-foreground">Drop Columns</label>
                    <Input
                      value={source.dropColumns}
                      onChange={(e) =>
                        setHfCustomSources((current) =>
                          current.map((row) =>
                            row.id === source.id ? { ...row, dropColumns: e.target.value } : row,
                          ),
                        )
                      }
                      placeholder="chat_template_kwargs"
                    />
                  </div>
                  <div className="space-y-1 md:col-span-2">
                    <label className="text-xs font-medium text-muted-foreground">
                      Rename Columns (JSON object)
                    </label>
                    <textarea
                      value={source.renameColumns}
                      onChange={(e) =>
                        setHfCustomSources((current) =>
                          current.map((row) =>
                            row.id === source.id ? { ...row, renameColumns: e.target.value } : row,
                          ),
                        )
                      }
                      placeholder='{"dataset":"source"}'
                      className="min-h-20 w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm text-foreground"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
          <Button
            type="button"
            variant="outline"
            onClick={() => setHfCustomSources((current) => [...current, createEmptyHfSource()])}
          >
            <Plus className="h-4 w-4" />
            Add Source
          </Button>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Output Path</label>
              <Input
                value={hfOutputPath}
                onChange={(e) => setHfOutputPath(e.target.value)}
                placeholder={`${getDefaultDatasetOutputDir()}/hf_blend_sft.jsonl`}
              />
            </div>
          </div>
          <Button onClick={handleComposeRecipe} disabled={isPending || isComposingRecipe}>
            {isComposingRecipe && <Loader2 className="h-4 w-4 animate-spin" />}
            {isComposingRecipe ? 'Composing...' : 'Compose HF Dataset'}
          </Button>
          <p className="text-xs text-muted-foreground">
            The job always runs from the source rows shown here.
          </p>
          {hfBlendJobStatus && (
            <div className="space-y-2 rounded-md border border-border/60 bg-secondary/20 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline">Job: {hfBlendJobStatus.job_id}</Badge>
                <Badge variant="outline">Status: {hfBlendJobStatus.status}</Badge>
                {typeof hfBlendJobStatus.progress?.percent_complete === 'number' && (
                  <Badge variant="outline">
                    Progress: {Math.round(hfBlendJobStatus.progress.percent_complete)}%
                  </Badge>
                )}
              </div>
              {hfBlendJobStatus.progress?.status_message && (
                <p className="text-xs text-muted-foreground">
                  {hfBlendJobStatus.progress.status_message}
                </p>
              )}
            </div>
          )}
          {hfGenerationSummary && (
            <div className="space-y-2 rounded-md border border-border/60 bg-secondary/20 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="success">Generated: {hfGenerationSummary.count} rows</Badge>
                <Badge variant="outline">
                  Saved: {hfGenerationSummary.outputPath.split(/[\\/]/).pop()}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground break-all">{hfGenerationSummary.outputPath}</p>
            </div>
          )}
          {hfComposeResult && (
            <pre className="text-xs bg-secondary/50 rounded p-3 overflow-auto max-h-48">
              {hfComposeResult}
            </pre>
          )}
        </CardContent>
      </Card>

      <VlmDatasetBuilder />

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
