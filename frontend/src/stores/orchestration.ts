import { create } from 'zustand'

interface OrchestrationStore {
  currentStep: number
  stepResults: Record<number, Record<string, unknown>>
  setStep: (step: number) => void
  setStepResult: (step: number, result: Record<string, unknown>) => void
  reset: () => void
}

export const useOrchestrationStore = create<OrchestrationStore>()((set) => ({
  currentStep: 0,
  stepResults: {},
  setStep: (step) => set({ currentStep: step }),
  setStepResult: (step, result) =>
    set((s) => ({ stepResults: { ...s.stepResults, [step]: result } })),
  reset: () => set({ currentStep: 0, stepResults: {} }),
}))
