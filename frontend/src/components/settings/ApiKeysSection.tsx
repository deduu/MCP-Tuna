import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Key, Eye, EyeOff, CheckCircle } from 'lucide-react'
import { toast } from 'sonner'
import { useSetHFToken, useSetupCheck } from '@/api/hooks/useSystemResources'

export function ApiKeysSection() {
  const [token, setToken] = useState('')
  const [showToken, setShowToken] = useState(false)
  const setupCheck = useSetupCheck()
  const setHFToken = useSetHFToken()

  const hfCheck = setupCheck.data?.checks?.find((c) => c.name === 'HF_TOKEN')
  const isConfigured = hfCheck?.status === 'pass'

  const handleSave = () => {
    if (!token.trim()) return
    setHFToken.mutate(token, {
      onSuccess: (data) => {
        toast.success(`HF token saved${data.username ? ` for ${data.username}` : ''}`)
        setToken('')
      },
      onError: (err) => toast.error(err.message),
    })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Key className="h-4 w-4" />
          API Keys
        </CardTitle>
        <CardDescription>Manage authentication tokens for external services</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">HuggingFace Token</label>
            {isConfigured && (
              <span className="flex items-center gap-1 text-xs text-green-600">
                <CheckCircle className="h-3 w-3" />
                Configured
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                type={showToken ? 'text' : 'password'}
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder={isConfigured ? 'Token already set (enter to update)' : 'hf_...'}
                className="pr-9"
                onKeyDown={(e) => e.key === 'Enter' && handleSave()}
              />
              <button
                type="button"
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                onClick={() => setShowToken((s) => !s)}
              >
                {showToken ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            <Button
              onClick={handleSave}
              disabled={!token.trim() || setHFToken.isPending}
              size="sm"
            >
              {setHFToken.isPending ? 'Saving...' : 'Save'}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Required for gated models (Llama, Mistral). Token is in-memory only and
            resets on gateway restart. For persistence, set <code>HF_TOKEN</code> in your <code>.env</code> file.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
