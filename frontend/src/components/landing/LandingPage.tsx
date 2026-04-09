import { startTransition, useEffect, useEffectEvent, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Link } from 'react-router'
import {
  ArrowRight,
  Bot,
  Boxes,
  ChartColumnIncreasing,
  Cloud,
  Cpu,
  Database,
  Fish,
  GitBranch,
  Layers3,
  LockKeyhole,
  Rocket,
  ServerCog,
  ShieldCheck,
  Sparkles,
  Workflow,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const GITHUB_URL = 'https://github.com/deduu/MCP-Tuna'

interface PlatformStat {
  label: string
  value: number
  suffix?: string
  description: string
}

const PLATFORM_STATS: PlatformStat[] = [
  {
    label: 'Tool Entry Points',
    value: 120,
    suffix: '+',
    description: 'Schema-driven MCP tools for generation, evaluation, training, hosting, and orchestration.',
  },
  {
    label: 'Gateway Namespaces',
    value: 17,
    description: 'Capability families that stay discoverable for both agents and human operators.',
  },
  {
    label: 'Control Plane Surfaces',
    value: 8,
    description: 'Tools, chat, datasets, pipeline, training, deployments, evaluation, and settings.',
  },
  {
    label: 'Commercial Tracks',
    value: 3,
    description: 'Open-source, managed cloud, and enterprise deployment modes from one product story.',
  },
] as const

const PLATFORM_PILLARS = [
  {
    icon: Workflow,
    title: 'One workflow spine',
    description: 'Move from raw documents to deployed adapters without stitching together notebooks, scripts, and custom wrappers.',
  },
  {
    icon: Bot,
    title: 'Built for agents and operators',
    description: 'Expose the same capabilities through MCP for coding agents and through a control plane for human teams.',
  },
  {
    icon: ShieldCheck,
    title: 'Portable by default',
    description: 'Keep the open-source core, self-host if needed, and layer managed GPUs or enterprise controls on top later.',
  },
] as const

const BUYER_SEGMENTS = [
  {
    icon: Sparkles,
    title: 'AI product teams',
    description: 'Customize open models on domain data without building a post-training platform from scratch.',
  },
  {
    icon: GitBranch,
    title: 'Agencies and studios',
    description: 'Run repeatable client workflows across dataset prep, SFT, evaluation, and deployment from one stack.',
  },
  {
    icon: ServerCog,
    title: 'Internal platform groups',
    description: 'Offer a standard post-training surface to multiple builders while keeping infrastructure choices flexible.',
  },
] as const

const OPERATING_MODES = [
  {
    badge: 'Open Source',
    title: 'Self-host the core stack',
    description: 'Use MCP Tuna on local GPUs or your own cloud when control and portability matter most.',
    bullets: ['GitHub-first distribution', 'Own your datasets and adapters', 'No forced managed runtime'],
    accent: 'from-sky-500/25 via-sky-500/5 to-transparent',
    icon: Layers3,
  },
  {
    badge: 'Managed Cloud',
    title: 'Pay for GPU-backed execution',
    description: 'Commercialize with managed training, evaluation, and deployment for users who do not have their own GPUs.',
    bullets: ['Usage-metered GPU workloads', 'Queueing and run retention', 'Fast path from upload to endpoint'],
    accent: 'from-emerald-500/25 via-emerald-500/5 to-transparent',
    icon: Cloud,
  },
  {
    badge: 'Enterprise',
    title: 'Dedicated or BYOC delivery',
    description: 'Offer private capacity, tenant isolation, governance, and procurement-friendly deployment models.',
    bullets: ['Dedicated GPU pools', 'Private networking and storage', 'Future SSO, RBAC, and compliance controls'],
    accent: 'from-amber-500/25 via-amber-500/5 to-transparent',
    icon: LockKeyhole,
  },
] as const

const WORKFLOWS = [
  {
    id: 'sft',
    eyebrow: 'SFT-first alpha',
    title: 'Domain fine-tuning without the glue code tax',
    summary:
      'Upload documents, generate or import training rows, clean them, benchmark the result, and ship an endpoint from one operating surface.',
    stages: [
      { name: 'Ingest', detail: 'PDFs, raw text, and Hugging Face datasets land in one workspace.', progress: 92 },
      { name: 'Prepare', detail: 'Cleaning, normalization, validation, and quality filtering stay traceable.', progress: 86 },
      { name: 'Train', detail: 'LoRA-first SFT stays front-and-center while preference methods remain guarded.', progress: 78 },
      { name: 'Deploy', detail: 'Publish as MCP or API endpoints and test the result against live prompts.', progress: 88 },
    ],
    metrics: [
      { label: 'Primary user', value: 'Applied AI teams' },
      { label: 'Recommended path', value: 'Dataset -> SFT -> Eval -> Deploy' },
      { label: 'Why it wins', value: 'One codebase from prep to runtime' },
    ],
    accent: 'from-sky-500/30 via-cyan-500/10 to-transparent',
  },
  {
    id: 'evaluation',
    eyebrow: 'Evaluation-first iteration',
    title: 'Treat quality as a first-class product surface',
    summary:
      'Run judge flows, compare checkpoints, export benchmark reports, and keep training decisions tied to measurable outcomes.',
    stages: [
      { name: 'Compare', detail: 'Pointwise and pairwise judging show whether the adapter is actually better.', progress: 91 },
      { name: 'Diagnose', detail: 'Trace weak datasets, weak prompts, and weak runs before spending more GPU time.', progress: 83 },
      { name: 'Benchmark', detail: 'Keep internal baselines frozen so regressions are obvious instead of anecdotal.', progress: 79 },
      { name: 'Ship', detail: 'Promote only the versions that clear your evaluation bar.', progress: 87 },
    ],
    metrics: [
      { label: 'Primary user', value: 'Platform and ML ops teams' },
      { label: 'Recommended path', value: 'Baseline -> Judge -> Compare -> Promote' },
      { label: 'Why it wins', value: 'Fewer blind retrains and cleaner release decisions' },
    ],
    accent: 'from-fuchsia-500/30 via-pink-500/10 to-transparent',
  },
  {
    id: 'managed',
    eyebrow: 'Commercial track',
    title: 'Managed GPUs for teams that do not own compute',
    summary:
      'Position MCP Tuna Cloud as the hosted execution layer on top of the open-source core, with GPU providers supplying elastic capacity.',
    stages: [
      { name: 'Acquire', detail: 'A public landing page captures users who want open-source flexibility or hosted execution.', progress: 89 },
      { name: 'Queue', detail: 'Managed jobs route into GPU-backed workers with storage, artifacts, and status tracking.', progress: 76 },
      { name: 'Meter', detail: 'Usage is expressed in GPU time, storage, model endpoints, and API activity.', progress: 68 },
      { name: 'Expand', detail: 'Upsell from self-serve cloud to team and enterprise deployment models.', progress: 80 },
    ],
    metrics: [
      { label: 'Primary user', value: 'Agencies and no-GPU teams' },
      { label: 'Recommended path', value: 'OSS -> Managed Cloud -> Enterprise' },
      { label: 'Why it wins', value: 'Workflow product, not commodity compute' },
    ],
    accent: 'from-emerald-500/30 via-teal-500/10 to-transparent',
  },
] as const

function useCountUp(target: number, durationMs = 1400) {
  const [value, setValue] = useState(0)

  useEffect(() => {
    let frame = 0
    const start = window.performance.now()

    const tick = (timestamp: number) => {
      const progress = Math.min(1, (timestamp - start) / durationMs)
      const eased = 1 - Math.pow(1 - progress, 3)
      setValue(Math.round(target * eased))

      if (progress < 1) {
        frame = window.requestAnimationFrame(tick)
      }
    }

    frame = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(frame)
  }, [durationMs, target])

  return value
}

function StatCard({
  label,
  value,
  suffix,
  description,
}: {
  label: string
  value: number
  suffix?: string
  description: string
}) {
  const animatedValue = useCountUp(value)

  return (
    <Card className="border-white/10 bg-white/4 shadow-[0_24px_80px_rgba(5,10,20,0.38)] backdrop-blur-xl">
      <CardContent className="p-5 pt-5">
        <p className="text-[11px] uppercase tracking-[0.32em] text-slate-400">{label}</p>
        <p className="mt-3 text-4xl font-semibold tracking-tight text-white">
          {animatedValue}
          {suffix ?? ''}
        </p>
        <p className="mt-3 text-sm leading-6 text-slate-300">{description}</p>
      </CardContent>
    </Card>
  )
}

export function LandingPage() {
  const [activeWorkflowIndex, setActiveWorkflowIndex] = useState(0)
  const activeWorkflow = WORKFLOWS[activeWorkflowIndex]

  const rotateWorkflow = useEffectEvent(() => {
    startTransition(() => {
      setActiveWorkflowIndex((current) => (current + 1) % WORKFLOWS.length)
    })
  })

  useEffect(() => {
    const interval = window.setInterval(() => rotateWorkflow(), 5200)
    return () => window.clearInterval(interval)
  }, [rotateWorkflow])

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#071019] text-white">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(34,197,94,0.16),_transparent_24%),radial-gradient(circle_at_top_right,_rgba(59,130,246,0.16),_transparent_28%),radial-gradient(circle_at_55%_35%,_rgba(236,72,153,0.14),_transparent_22%),linear-gradient(180deg,_rgba(6,10,18,0.35)_0%,_rgba(6,10,18,0.9)_68%,_#071019_100%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.08)_1px,transparent_1px)] bg-[size:88px_88px] [mask-image:radial-gradient(circle_at_center,black,transparent_82%)]" />
      </div>

      <header className="sticky top-0 z-30 border-b border-white/8 bg-[#071019]/72 backdrop-blur-xl">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-5 py-4 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-sky-400/30 bg-sky-400/10 text-sky-300 shadow-[0_12px_32px_rgba(56,189,248,0.18)]">
              <Fish className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold tracking-tight">MCP Tuna</p>
              <p className="text-xs uppercase tracking-[0.26em] text-slate-400">Post-Training Platform</p>
            </div>
          </div>

          <nav className="hidden items-center gap-7 text-sm text-slate-300 lg:flex">
            <a href="#platform" className="transition-colors hover:text-white">Platform</a>
            <a href="#workflow" className="transition-colors hover:text-white">Workflow</a>
            <a href="#commercial" className="transition-colors hover:text-white">Commercialization</a>
            <a href="#audience" className="transition-colors hover:text-white">Positioning</a>
          </nav>

          <div className="flex items-center gap-3">
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
              className={cn(
                buttonVariants({ variant: 'outline', size: 'sm' }),
                'hidden rounded-full border-white/14 bg-white/4 px-4 text-slate-100 hover:bg-white/10 sm:inline-flex',
              )}
            >
              View GitHub
            </a>
            <Link
              to="/dashboard"
              className={cn(buttonVariants({ size: 'sm' }), 'rounded-full px-4 shadow-[0_10px_28px_rgba(59,130,246,0.32)]')}
            >
              Open Control Plane
            </Link>
          </div>
        </div>
      </header>

      <main className="relative">
        <section className="mx-auto flex w-full max-w-7xl flex-col gap-10 px-5 pb-20 pt-16 lg:px-8 lg:pb-28 lg:pt-20">
          <div className="grid gap-8 lg:grid-cols-[minmax(0,1.05fr)_minmax(420px,0.95fr)] lg:items-center">
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, ease: 'easeOut' }}
              className="space-y-7"
            >
              <Badge className="w-fit border border-emerald-400/25 bg-emerald-400/10 px-3 py-1 text-emerald-200">
                Open-source core. Managed GPUs when users need them.
              </Badge>

              <div className="space-y-5">
                <h1 className="max-w-4xl text-5xl font-semibold leading-[1.02] tracking-[-0.04em] text-white sm:text-6xl lg:text-7xl">
                  The MCP-native stack for turning model customization into a product.
                </h1>
                <p className="max-w-2xl text-lg leading-8 text-slate-300 sm:text-xl">
                  MCP Tuna combines dataset generation, evaluation, fine-tuning, deployment, and operator tooling
                  in one workflow spine. Ship it as open source, then monetize managed GPU-backed execution and
                  enterprise delivery on top.
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <Link
                  to="/dashboard"
                  className={cn(
                    buttonVariants({ size: 'lg' }),
                    'h-12 rounded-full px-6 text-sm shadow-[0_16px_40px_rgba(59,130,246,0.35)]',
                  )}
                >
                  Explore The Product
                  <ArrowRight className="h-4 w-4" />
                </Link>
                <a
                  href="#commercial"
                  className={cn(
                    buttonVariants({ variant: 'outline', size: 'lg' }),
                    'h-12 rounded-full border-white/14 bg-white/4 px-6 text-sm text-white hover:bg-white/10',
                  )}
                >
                  See The SaaS Model
                </a>
              </div>

              <div className="flex flex-wrap gap-2 text-xs uppercase tracking-[0.24em] text-slate-400">
                <span className="rounded-full border border-white/10 bg-white/4 px-3 py-1.5">MCP gateway + split servers</span>
                <span className="rounded-full border border-white/10 bg-white/4 px-3 py-1.5">SFT-first product path</span>
                <span className="rounded-full border border-white/10 bg-white/4 px-3 py-1.5">Managed cloud upsell</span>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 28 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.08, ease: 'easeOut' }}
              className="relative overflow-hidden rounded-[28px] border border-white/10 bg-white/5 p-5 shadow-[0_34px_120px_rgba(5,10,20,0.45)] backdrop-blur-2xl"
            >
              <div className={cn('absolute inset-0 bg-gradient-to-br', activeWorkflow.accent)} />
              <div className="relative space-y-5">
                <div className="flex items-center justify-between gap-3 rounded-2xl border border-white/10 bg-[#071019]/72 px-4 py-3">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.28em] text-slate-400">Active Narrative</p>
                    <p className="mt-1 text-sm font-medium text-white">{activeWorkflow.eyebrow}</p>
                  </div>
                  <Badge variant="secondary" className="border border-white/10 bg-white/10 text-slate-100">
                    Dynamic Workflow Preview
                  </Badge>
                </div>

                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeWorkflow.id}
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -16 }}
                    transition={{ duration: 0.35, ease: 'easeOut' }}
                    className="space-y-5"
                  >
                    <div className="space-y-2">
                      <p className="text-xs uppercase tracking-[0.24em] text-slate-300">{activeWorkflow.eyebrow}</p>
                      <h2 className="text-3xl font-semibold tracking-[-0.03em] text-white">{activeWorkflow.title}</h2>
                      <p className="max-w-2xl text-sm leading-7 text-slate-300">{activeWorkflow.summary}</p>
                    </div>

                    <div className="grid gap-3">
                      {activeWorkflow.stages.map((stage) => (
                        <div key={stage.name} className="rounded-2xl border border-white/10 bg-[#08111c]/78 p-4">
                          <div className="flex items-center justify-between gap-4">
                            <div>
                              <p className="text-sm font-medium text-white">{stage.name}</p>
                              <p className="mt-1 text-sm text-slate-400">{stage.detail}</p>
                            </div>
                            <div className="text-right">
                              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Readiness</p>
                              <p className="mt-1 text-lg font-semibold text-white">{stage.progress}%</p>
                            </div>
                          </div>
                          <div className="mt-4 h-2 rounded-full bg-white/8">
                            <motion.div
                              className="h-full rounded-full bg-gradient-to-r from-sky-400 via-cyan-300 to-emerald-300"
                              initial={{ width: 0 }}
                              animate={{ width: `${stage.progress}%` }}
                              transition={{ duration: 0.65, ease: 'easeOut' }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="grid gap-3 md:grid-cols-3">
                      {activeWorkflow.metrics.map((metric) => (
                        <div key={metric.label} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                          <p className="text-[11px] uppercase tracking-[0.28em] text-slate-400">{metric.label}</p>
                          <p className="mt-2 text-sm font-medium leading-6 text-white">{metric.value}</p>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                </AnimatePresence>

                <div className="flex flex-wrap gap-2">
                  {WORKFLOWS.map((workflow, index) => (
                    <button
                      key={workflow.id}
                      type="button"
                      onClick={() => setActiveWorkflowIndex(index)}
                      className={cn(
                        'rounded-full border px-3 py-2 text-xs font-medium tracking-[0.2em] uppercase transition-all',
                        index === activeWorkflowIndex
                          ? 'border-sky-300/40 bg-sky-300/12 text-sky-100'
                          : 'border-white/10 bg-white/4 text-slate-400 hover:border-white/18 hover:text-white',
                      )}
                    >
                      {workflow.eyebrow}
                    </button>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {PLATFORM_STATS.map((stat) => (
              <StatCard
                key={stat.label}
                label={stat.label}
                value={stat.value}
                suffix={stat.suffix}
                description={stat.description}
              />
            ))}
          </div>
        </section>

        <section id="platform" className="mx-auto w-full max-w-7xl px-5 pb-20 lg:px-8">
          <div className="mb-8 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Platform Thesis</p>
              <h2 className="max-w-3xl text-3xl font-semibold tracking-[-0.03em] text-white sm:text-4xl">
                MCP Tuna should be sold as workflow infrastructure, not as generic GPU rental.
              </h2>
            </div>
            <p className="max-w-xl text-sm leading-7 text-slate-300">
              The commercial story is stronger when the open-source stack stays credible on its own and the managed
              offer removes operational friction for teams that lack GPUs, MLOps, or deployment bandwidth.
            </p>
          </div>

          <div className="grid gap-4 lg:grid-cols-3">
            {PLATFORM_PILLARS.map((pillar, index) => (
              <motion.div
                key={pillar.title}
                initial={{ opacity: 0, y: 18 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.4 }}
                transition={{ duration: 0.45, delay: index * 0.08 }}
              >
                <Card className="h-full border-white/10 bg-white/5 backdrop-blur-xl">
                  <CardHeader className="space-y-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/8 text-sky-200">
                      <pillar.icon className="h-5 w-5" />
                    </div>
                    <div className="space-y-2">
                      <CardTitle className="text-xl text-white">{pillar.title}</CardTitle>
                      <CardDescription className="text-sm leading-7 text-slate-300">{pillar.description}</CardDescription>
                    </div>
                  </CardHeader>
                </Card>
              </motion.div>
            ))}
          </div>
        </section>

        <section id="workflow" className="mx-auto w-full max-w-7xl px-5 pb-20 lg:px-8">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,0.86fr)_minmax(0,1.14fr)]">
            <Card className="overflow-hidden border-white/10 bg-white/5 backdrop-blur-xl">
              <CardHeader className="space-y-3">
                <Badge variant="outline" className="w-fit border-white/12 text-slate-300">
                  Go-to-Market Narrative
                </Badge>
                <CardTitle className="text-3xl tracking-[-0.03em] text-white">
                  Open-source distribution pulls users in. Managed execution becomes the paid layer.
                </CardTitle>
                <CardDescription className="max-w-xl text-sm leading-7 text-slate-300">
                  That split keeps the GitHub story authentic while making the SaaS motion obvious: users can start
                  free, then pay when they need reliable GPU-backed runs, team workflows, and production delivery.
                </CardDescription>
              </CardHeader>
            </Card>

            <div className="grid gap-4 md:grid-cols-3">
              {BUYER_SEGMENTS.map((segment, index) => (
                <motion.div
                  key={segment.title}
                  initial={{ opacity: 0, y: 18 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, amount: 0.5 }}
                  transition={{ duration: 0.42, delay: index * 0.08 }}
                >
                  <Card className="h-full border-white/10 bg-[#09131f]/92 backdrop-blur-xl">
                    <CardHeader className="space-y-4">
                      <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/8 text-emerald-200">
                        <segment.icon className="h-5 w-5" />
                      </div>
                      <div className="space-y-2">
                        <CardTitle className="text-lg text-white">{segment.title}</CardTitle>
                        <CardDescription className="text-sm leading-7 text-slate-300">{segment.description}</CardDescription>
                      </div>
                    </CardHeader>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section id="commercial" className="mx-auto w-full max-w-7xl px-5 pb-24 lg:px-8">
          <div className="mb-8 space-y-3">
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Commercialization Model</p>
            <h2 className="max-w-3xl text-3xl font-semibold tracking-[-0.03em] text-white sm:text-4xl">
              Three clear operating models let the product stay open while the business captures paid execution.
            </h2>
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            {OPERATING_MODES.map((mode, index) => (
              <motion.div
                key={mode.title}
                initial={{ opacity: 0, y: 18 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.35 }}
                transition={{ duration: 0.42, delay: index * 0.08 }}
              >
                <Card className="relative h-full overflow-hidden border-white/10 bg-[#08111c]/92 backdrop-blur-xl">
                  <div className={cn('absolute inset-x-0 top-0 h-40 bg-gradient-to-b', mode.accent)} />
                  <CardHeader className="relative space-y-4">
                    <div className="flex items-start justify-between gap-4">
                      <Badge variant="secondary" className="border border-white/10 bg-white/10 text-slate-100">
                        {mode.badge}
                      </Badge>
                      <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/8 text-white">
                        <mode.icon className="h-5 w-5" />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <CardTitle className="text-2xl text-white">{mode.title}</CardTitle>
                      <CardDescription className="text-sm leading-7 text-slate-300">{mode.description}</CardDescription>
                    </div>
                  </CardHeader>
                  <CardContent className="relative space-y-3 pt-0">
                    {mode.bullets.map((bullet) => (
                      <div key={bullet} className="flex items-start gap-3 rounded-2xl border border-white/8 bg-white/4 px-4 py-3">
                        <div className="mt-1 h-2 w-2 rounded-full bg-sky-300" />
                        <p className="text-sm leading-6 text-slate-200">{bullet}</p>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          <div className="mt-8 grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
            <Card className="border-white/10 bg-white/5 backdrop-blur-xl">
              <CardHeader className="space-y-3">
                <Badge variant="outline" className="w-fit border-white/12 text-slate-300">
                  Recommended Wedge
                </Badge>
                <CardTitle className="text-2xl text-white">Start with SFT and evaluation for teams that lack GPU capacity.</CardTitle>
                <CardDescription className="text-sm leading-7 text-slate-300">
                  This is the most supportable early offer. It matches the current product maturity, lines up with the
                  existing alpha roadmap, and avoids over-promising mature preference-tuning or enterprise-grade SaaS
                  controls before the platform is ready.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-white/10 bg-[#09131f]/95 backdrop-blur-xl">
              <CardHeader className="space-y-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/8 text-amber-200">
                  <ChartColumnIncreasing className="h-5 w-5" />
                </div>
                <div className="space-y-2">
                  <CardTitle className="text-2xl text-white">Revenue logic</CardTitle>
                  <CardDescription className="text-sm leading-7 text-slate-300">
                    Open source builds trust and reach. Managed cloud captures GPU-backed usage. Enterprise captures
                    dedicated capacity, procurement, and private deployment requirements.
                  </CardDescription>
                </div>
              </CardHeader>
            </Card>
          </div>
        </section>

        <section id="audience" className="border-t border-white/8 bg-[#060d16]/78">
          <div className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-5 py-16 lg:flex-row lg:items-end lg:justify-between lg:px-8">
            <div className="space-y-4">
              <Badge variant="outline" className="w-fit border-white/12 text-slate-300">
                Positioning
              </Badge>
              <h2 className="max-w-3xl text-3xl font-semibold tracking-[-0.03em] text-white sm:text-4xl">
                Position MCP Tuna as the post-training product layer between model experimentation and production operations.
              </h2>
              <p className="max-w-3xl text-sm leading-7 text-slate-300">
                The message should stay simple: open models become more useful when dataset prep, SFT, evaluation,
                deployment, and agent access share one reliable operating surface.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <Link
                to="/dashboard"
                className={cn(
                  buttonVariants({ size: 'lg' }),
                  'h-12 rounded-full px-6 shadow-[0_16px_40px_rgba(59,130,246,0.35)]',
                )}
              >
                Open Workspace
                <ArrowRight className="h-4 w-4" />
              </Link>
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noreferrer"
                className={cn(
                  buttonVariants({ variant: 'outline', size: 'lg' }),
                  'h-12 rounded-full border-white/14 bg-white/4 px-6 text-white hover:bg-white/10',
                )}
              >
                View Open-Source Repo
              </a>
            </div>
          </div>

          <div className="mx-auto grid w-full max-w-7xl gap-4 px-5 pb-20 lg:grid-cols-4 lg:px-8">
            {[
              { icon: Database, label: 'Data', text: 'Import, generate, clean, normalize, and score training data.' },
              { icon: Cpu, label: 'Training', text: 'Keep SFT as the supported path while advanced methods stay clearly gated.' },
              { icon: Rocket, label: 'Runtime', text: 'Deploy adapters as live APIs or MCP servers with test loops attached.' },
              { icon: Boxes, label: 'Agent Access', text: 'Expose the same capability surface to coding agents and operators.' },
            ].map((item) => (
              <Card key={item.label} className="border-white/10 bg-white/5 backdrop-blur-xl">
                <CardContent className="space-y-4 p-5 pt-5">
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/8 text-sky-200">
                    <item.icon className="h-5 w-5" />
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm font-semibold text-white">{item.label}</p>
                    <p className="text-sm leading-7 text-slate-300">{item.text}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}
