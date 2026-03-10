import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Key, Eye, EyeOff, CheckCircle, Link2 } from 'lucide-react'
import { toast } from 'sonner'
import { useSetHFToken, useSetRuntimeEnv, useSetupCheck, useSystemConfig } from '@/api/hooks/useSystemResources'

function statusVariant(configured: boolean) {
  return configured ? 'success' : 'warning'
}

function ProviderField({
  label,
  placeholder,
  value,
  configured,
  showValue,
  canToggle,
  onToggle,
  onChange,
  onSave,
  saveLabel,
  disabled,
  hint,
}: {
  label: string
  placeholder: string
  value: string
  configured: boolean
  showValue?: boolean
  canToggle?: boolean
  onToggle?: () => void
  onChange: (value: string) => void
  onSave: () => void
  saveLabel: string
  disabled: boolean
  hint: string
}) {
  return (
    <div className="space-y-2 rounded-xl border border-border/60 bg-secondary/10 p-4">
      <div className="flex items-center justify-between gap-3">
        <label className="text-sm font-medium">{label}</label>
        <Badge variant={statusVariant(configured)}>
          {configured ? 'Configured' : 'Optional'}
        </Badge>
      </div>
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Input
            type={showValue ? 'text' : 'password'}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
            className={canToggle ? 'pr-9' : undefined}
            onKeyDown={(e) => e.key === 'Enter' && onSave()}
          />
          {canToggle && onToggle ? (
            <button
              type="button"
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              onClick={onToggle}
            >
              {showValue ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          ) : null}
        </div>
        <Button onClick={onSave} disabled={disabled} size="sm">
          {saveLabel}
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  )
}

export function ApiKeysSection() {
  const setupCheck = useSetupCheck()
  const { data: config } = useSystemConfig()
  const setRuntimeEnv = useSetRuntimeEnv()
  const setHFTokenMutation = useSetHFToken()

  const [showOpenAIKey, setShowOpenAIKey] = useState(false)
  const [showAnthropicKey, setShowAnthropicKey] = useState(false)
  const [showGoogleKey, setShowGoogleKey] = useState(false)
  const [showHFToken, setShowHFToken] = useState(false)

  const [openAIKey, setOpenAIKey] = useState('')
  const [openAIBase, setOpenAIBase] = useState('')
  const [anthropicKey, setAnthropicKey] = useState('')
  const [anthropicBase, setAnthropicBase] = useState('')
  const [googleKey, setGoogleKey] = useState('')
  const [hfToken, setHFTokenValue] = useState('')

  const checks = setupCheck.data?.checks ?? []
  const env = config?.env ?? {}

  const isConfigured = (name: string) => checks.find((c) => c.name === name)?.status === 'pass'

  const saveEnv = (key: string, value: string, successLabel: string) => {
    setRuntimeEnv.mutate(
      { key, value },
      {
        onSuccess: () => toast.success(successLabel),
        onError: (err) => toast.error(err.message),
      },
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Key className="h-4 w-4" />
          Provider Settings
        </CardTitle>
        <CardDescription>
          Configure provider credentials and base URLs for the current gateway session
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/10 p-4 text-sm text-amber-100">
          Runtime changes apply to the current gateway process only. Persist them in your
          <code className="mx-1">.env</code>
          file if you want them to survive a restart.
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <ProviderField
            label="OpenAI API Key"
            placeholder={isConfigured('OpenAI API Key') ? 'Already configured. Enter a new key to replace it.' : 'sk-...'}
            value={openAIKey}
            configured={isConfigured('OpenAI API Key')}
            showValue={showOpenAIKey}
            canToggle
            onToggle={() => setShowOpenAIKey((s) => !s)}
            onChange={setOpenAIKey}
            onSave={() => saveEnv('OPENAI_API_KEY', openAIKey, 'OpenAI API key updated')}
            saveLabel={setRuntimeEnv.isPending ? 'Saving...' : 'Save'}
            disabled={!openAIKey.trim() || setRuntimeEnv.isPending}
            hint="Needed for GPT models and OpenAI-compatible providers."
          />

          <ProviderField
            label="OpenAI Base URL"
            placeholder={env.OPENAI_API_BASE ?? 'https://api.openai.com/v1'}
            value={openAIBase}
            configured={isConfigured('OpenAI Base URL')}
            showValue
            onChange={setOpenAIBase}
            onSave={() => saveEnv('OPENAI_API_BASE', openAIBase, 'OpenAI base URL updated')}
            saveLabel={setRuntimeEnv.isPending ? 'Saving...' : 'Save'}
            disabled={!openAIBase.trim() || setRuntimeEnv.isPending}
            hint="Optional. Useful for OpenRouter, local gateways, or any OpenAI-compatible endpoint."
          />

          <ProviderField
            label="Anthropic API Key"
            placeholder={isConfigured('Anthropic API Key') ? 'Already configured. Enter a new key to replace it.' : 'sk-ant-...'}
            value={anthropicKey}
            configured={isConfigured('Anthropic API Key')}
            showValue={showAnthropicKey}
            canToggle
            onToggle={() => setShowAnthropicKey((s) => !s)}
            onChange={setAnthropicKey}
            onSave={() => saveEnv('ANTHROPIC_API_KEY', anthropicKey, 'Anthropic API key updated')}
            saveLabel={setRuntimeEnv.isPending ? 'Saving...' : 'Save'}
            disabled={!anthropicKey.trim() || setRuntimeEnv.isPending}
            hint="Needed for Claude models."
          />

          <ProviderField
            label="Anthropic Base URL"
            placeholder={env.ANTHROPIC_API_BASE ?? 'https://api.anthropic.com'}
            value={anthropicBase}
            configured={isConfigured('Anthropic Base URL')}
            showValue
            onChange={setAnthropicBase}
            onSave={() => saveEnv('ANTHROPIC_API_BASE', anthropicBase, 'Anthropic base URL updated')}
            saveLabel={setRuntimeEnv.isPending ? 'Saving...' : 'Save'}
            disabled={!anthropicBase.trim() || setRuntimeEnv.isPending}
            hint="Optional. Use for proxies or Anthropic-compatible gateways."
          />

          <ProviderField
            label="Google API Key"
            placeholder={isConfigured('Google API Key') ? 'Already configured. Enter a new key to replace it.' : 'AIza...'}
            value={googleKey}
            configured={isConfigured('Google API Key')}
            showValue={showGoogleKey}
            canToggle
            onToggle={() => setShowGoogleKey((s) => !s)}
            onChange={setGoogleKey}
            onSave={() => saveEnv('GOOGLE_API_KEY', googleKey, 'Google API key updated')}
            saveLabel={setRuntimeEnv.isPending ? 'Saving...' : 'Save'}
            disabled={!googleKey.trim() || setRuntimeEnv.isPending}
            hint="Needed for Gemini models."
          />

          <ProviderField
            label="HuggingFace Token"
            placeholder={isConfigured('HF Token') ? 'Already configured. Enter a new token to replace it.' : 'hf_...'}
            value={hfToken}
            configured={isConfigured('HF Token')}
            showValue={showHFToken}
            canToggle
            onToggle={() => setShowHFToken((s) => !s)}
            onChange={setHFTokenValue}
            onSave={() => {
              if (!hfToken.trim()) return
              setHFTokenMutation.mutate(hfToken, {
                onSuccess: (data) => {
                  toast.success(`HF token saved${data.username ? ` for ${data.username}` : ''}`)
                  setHFTokenValue('')
                },
                onError: (err) => toast.error(err.message),
              })
            }}
            saveLabel={setHFTokenMutation.isPending ? 'Saving...' : 'Save'}
            disabled={!hfToken.trim() || setHFTokenMutation.isPending}
            hint="Optional, but needed for gated model downloads and push_to_hub."
          />
        </div>

        <div className="rounded-xl border border-border/60 bg-secondary/10 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium">
            <CheckCircle className="h-4 w-4 text-emerald-400" />
            Current Provider Snapshot
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant={statusVariant(isConfigured('LLM API Provider'))}>LLM Provider</Badge>
            <Badge variant={statusVariant(isConfigured('OpenAI API Key'))}>OpenAI</Badge>
            <Badge variant={statusVariant(isConfigured('Anthropic API Key'))}>Anthropic</Badge>
            <Badge variant={statusVariant(isConfigured('Google API Key'))}>Gemini</Badge>
            <Badge variant={statusVariant(isConfigured('HF Token'))}>HuggingFace</Badge>
            {env.OPENAI_API_BASE && (
              <Badge variant="outline" className="gap-1.5">
                <Link2 className="h-3 w-3" />
                OpenAI Base URL
              </Badge>
            )}
            {env.ANTHROPIC_API_BASE && (
              <Badge variant="outline" className="gap-1.5">
                <Link2 className="h-3 w-3" />
                Anthropic Base URL
              </Badge>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
