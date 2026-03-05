import { RotateCcw } from 'lucide-react'
import { useOrchestrationStore } from '@/stores/orchestration'
import { Button } from '@/components/ui/button'
import { StepIndicator } from './StepIndicator'
import { OrchestrationStep } from './OrchestrationStep'

const ORCH_STEPS = [
  'Generate Problems',
  'Collect Trajectories',
  'Score Trajectories',
  'Format Dataset',
]

export function OrchestrationTab() {
  const { currentStep, setStep, setStepResult, reset } = useOrchestrationStore()

  const isComplete = currentStep >= ORCH_STEPS.length

  function handleStepComplete(result: Record<string, unknown>) {
    setStepResult(currentStep, result)
    setStep(currentStep + 1)
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <StepIndicator steps={ORCH_STEPS} currentStep={currentStep} />
        <Button variant="ghost" size="sm" onClick={reset} className="gap-1 text-xs shrink-0 ml-4">
          <RotateCcw className="h-3.5 w-3.5" />
          Reset
        </Button>
      </div>

      {isComplete ? (
        <div className="text-center py-12">
          <p className="text-lg font-semibold text-emerald-400 mb-2">
            Orchestration Complete
          </p>
          <p className="text-sm text-muted-foreground mb-4">
            All 4 steps finished. Your dataset is ready for training.
          </p>
          <Button variant="secondary" onClick={reset}>
            Start Over
          </Button>
        </div>
      ) : (
        <OrchestrationStep
          key={currentStep}
          stepIndex={currentStep}
          onComplete={handleStepComplete}
        />
      )}
    </div>
  )
}
