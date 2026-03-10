import { create } from 'zustand'

export interface OrchestrationSettings {
  domainDescription: string
  numProblems: number
  nPerProblem: number
  outputFormat: 'sft' | 'dpo' | 'grpo'
  costBudget: number
  timeBudget: number
  outputDir: string
  baseModel: string
  numEpochs: number
  deploy: boolean
  deployPort: number
}

export const DEFAULT_ORCHESTRATION_SETTINGS: OrchestrationSettings = {
  domainDescription: 'general assistant workflows',
  numProblems: 10,
  nPerProblem: 3,
  outputFormat: 'sft',
  costBudget: 1,
  timeBudget: 60,
  outputDir: './output/orchestrator',
  baseModel: '',
  numEpochs: 3,
  deploy: false,
  deployPort: 8002,
}

interface OrchestrationStore {
  currentStep: number
  stepResults: Record<number, Record<string, unknown>>
  settings: OrchestrationSettings
  setStep: (step: number) => void
  setStepResult: (step: number, result: Record<string, unknown>) => void
  setSettings: (patch: Partial<OrchestrationSettings>) => void
  reset: () => void
}

export const useOrchestrationStore = create<OrchestrationStore>()((set) => ({
  currentStep: 0,
  stepResults: {},
  settings: DEFAULT_ORCHESTRATION_SETTINGS,
  setStep: (step) => set({ currentStep: step }),
  setStepResult: (step, result) =>
    set((s) => ({ stepResults: { ...s.stepResults, [step]: result } })),
  setSettings: (patch) =>
    set((s) => ({ settings: { ...s.settings, ...patch } })),
  reset: () => set({
    currentStep: 0,
    stepResults: {},
    settings: DEFAULT_ORCHESTRATION_SETTINGS,
  }),
}))
