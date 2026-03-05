import { Card, CardContent } from '@/components/ui/card'
import { Construction } from 'lucide-react'

export function PlaceholderPage({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex items-center justify-center h-[60vh]">
      <Card className="max-w-md w-full">
        <CardContent className="p-8 text-center space-y-3">
          <Construction className="h-10 w-10 mx-auto text-muted-foreground" />
          <h2 className="font-semibold text-lg">{title}</h2>
          <p className="text-sm text-muted-foreground">{description}</p>
        </CardContent>
      </Card>
    </div>
  )
}
