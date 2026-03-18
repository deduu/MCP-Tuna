import { mcpCall } from '@/api/client'

export type UploadCategory = 'documents' | 'images'

export interface UploadedAsset {
  filePath: string
  fileName: string
  previewUrl?: string
}

function sanitizeBaseName(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'upload'
}

export async function toBase64(file: File): Promise<string> {
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

export async function uploadAsset(
  file: File,
  category: UploadCategory,
): Promise<UploadedAsset> {
  const dot = file.name.lastIndexOf('.')
  const baseName = dot >= 0 ? file.name.slice(0, dot) : file.name
  const extension = dot >= 0 ? file.name.slice(dot) : ''
  const safeBaseName = sanitizeBaseName(baseName)
  const serverFilename = `${category}/${crypto.randomUUID()}_${safeBaseName}${extension}`
  const contentBase64 = await toBase64(file)
  const uploaded = await mcpCall<Record<string, unknown>>('file.upload', {
    filename: serverFilename,
    content_base64: contentBase64,
  })
  const filePath = typeof uploaded.file_path === 'string' ? uploaded.file_path : ''
  if (!filePath.trim()) {
    throw new Error('Upload succeeded but no server file path was returned')
  }

  return {
    filePath,
    fileName: file.name,
    previewUrl: category === 'images' ? URL.createObjectURL(file) : undefined,
  }
}
