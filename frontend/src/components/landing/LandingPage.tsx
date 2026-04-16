import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link } from 'react-router'
import {
  ArrowRight,
  Fish,
  Bot,
  Zap,
  CheckCircle2,
  LockKeyhole,
  Cloud,
  Layers3,
  Activity
} from 'lucide-react'
import { buttonVariants } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const GITHUB_URL = 'https://github.com/deduu/MCP-Tuna'

const AUDIENCES = [
  'Data Science Teams',
  'AI Product Engineers',
  'Agent Developers',
  'Machine Learning Ops'
]

function HeroSection() {
  const [audienceIndex, setAudienceIndex] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setAudienceIndex((prev) => (prev + 1) % AUDIENCES.length)
    }, 2500)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="relative pt-32 pb-20 flex flex-col items-center text-center px-5">
      <motion.h1
        className="max-w-5xl text-5xl sm:text-7xl font-bold tracking-tight text-white leading-[1.1]"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        From raw data to a deployed model in minutes.
      </motion.h1>

      <div className="mt-8 text-4xl sm:text-6xl font-semibold tracking-tight text-slate-400 h-[80px] flex items-center justify-center overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={audienceIndex}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.4 }}
            className="bg-clip-text text-transparent bg-gradient-to-r from-sky-400 to-emerald-400 pb-2"
          >
            # {AUDIENCES[audienceIndex]}
          </motion.div>
        </AnimatePresence>
      </div>

      <motion.p
        className="mt-6 text-xl sm:text-2xl text-slate-300 max-w-3xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.6 }}
      >
        Build beautiful, evaluated agent and model workflows fast without custom glue code. 10x faster than manual scripting.
      </motion.p>

      <motion.div
        className="mt-10 flex flex-wrap justify-center gap-4"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.6 }}
      >
        <Link
          to="/dashboard"
          className={cn(
            buttonVariants({ size: 'lg' }),
            'h-14 rounded-full px-8 text-base shadow-[0_16px_40px_rgba(59,130,246,0.35)]',
          )}
        >
          Explore Product
          <ArrowRight className="h-4 w-4 ml-2" />
        </Link>
      </motion.div>
    </div>
  )
}

function MockTerminal() {
  return (
    <div className="relative w-full max-w-5xl mx-auto mt-16 mb-32 group">
      <div className="absolute -inset-1 rounded-2xl bg-gradient-to-r from-sky-500/20 to-emerald-500/20 blur-xl opacity-50 group-hover:opacity-100 transition-opacity duration-700"></div>
      <div className="relative rounded-2xl border border-white/10 bg-[#060a0f] shadow-2xl overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10 bg-[#0b121b]">
          <div className="w-3 h-3 rounded-full bg-red-400/80"></div>
          <div className="w-3 h-3 rounded-full bg-amber-400/80"></div>
          <div className="w-3 h-3 rounded-full bg-emerald-400/80"></div>
          <div className="ml-4 text-xs font-mono text-slate-500">mcp-tuna-node</div>
        </div>
        <div className="p-6 font-mono text-sm leading-relaxed overflow-hidden h-80 space-y-2 relative">
          <motion.div initial={{opacity: 0, x:-10}} animate={{opacity: 1, x:0}} transition={{delay: 0.5}}>
            <span className="text-emerald-400">➜</span> <span className="text-sky-300">~</span> <span className="text-white">mcp-tuna process --data source.csv</span>
          </motion.div>
          <motion.div className="text-slate-400" initial={{opacity: 0}} animate={{opacity: 1}} transition={{delay: 1.2}}>
            Parsed 14,020 rows. Cleaning and formatting...
          </motion.div>
          <motion.div initial={{opacity: 0, x:-10}} animate={{opacity: 1, x:0}} transition={{delay: 2.5}}>
            <span className="text-emerald-400">➜</span> <span className="text-sky-300">~</span> <span className="text-white">mcp-tuna finetune --model llama-3</span>
          </motion.div>
          <motion.div className="text-slate-400" initial={{opacity: 0}} animate={{opacity: 1}} transition={{delay: 3.2}}>
            Initializing QLoRA adapters... Loss: 0.842
          </motion.div>
          <motion.div initial={{opacity: 0, x:-10}} animate={{opacity: 1, x:0}} transition={{delay: 4.5}}>
            <span className="text-emerald-400">➜</span> <span className="text-sky-300">~</span> <span className="text-white">mcp-tuna run evaluate</span>
          </motion.div>
          <motion.div className="text-slate-400" initial={{opacity: 0}} animate={{opacity: 1}} transition={{delay: 5.2}}>
             <span className="text-sky-400">Result:</span> Checkpoint outperforms baseline by 18%
          </motion.div>
          <motion.div initial={{opacity: 0, x:-10}} animate={{opacity: 1, x:0}} transition={{delay: 6.5}}>
            <span className="text-emerald-400">➜</span> <span className="text-sky-300">~</span> <span className="text-white">mcp-tuna deploy --endpoint api</span>
          </motion.div>
          <motion.div className="text-emerald-300 font-semibold" initial={{opacity: 0}} animate={{opacity: 1}} transition={{delay: 7.2}}>
            ✓ Model successfully hosted at https://api.mcp-tuna.local/v1
          </motion.div>
          
          <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-[#060a0f] to-transparent pointer-events-none"></div>
        </div>
      </div>
    </div>
  )
}

function ValueProps() {
  const props = [
    {
      icon: Zap,
      title: 'Abstracted Fine-Tuning & Deployment.',
      desc: 'Skip the Hugging Face boilerplate. Use our clean UI to seamlessly process data, fine-tune models, evaluate quality, and deploy endpoints without writing glue code.'
    },
    {
      icon: Bot,
      title: 'Agentic by design.',
      desc: 'Built on the Model Context Protocol. Connect your favorite AI agents to execute end-to-end fine-tuning tasks purely through natural language.'
    },
    {
      icon: Activity,
      title: 'Rigorous Observability.',
      desc: 'Compare generation quality pointwise and pairwise before making a single release. Real metrics and deep evaluations over subjective vibes.'
    }
  ]

  return (
    <div className="w-full max-w-6xl mx-auto grid gap-8 md:grid-cols-3 px-5 mb-32">
      {props.map((p, i) => (
        <motion.div 
          key={i} 
          className="p-6 rounded-2xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ delay: i * 0.1, duration: 0.5 }}
        >
          <div className="w-12 h-12 rounded-xl bg-sky-500/10 text-sky-400 flex items-center justify-center mb-6">
            <p.icon className="w-6 h-6" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-3">{p.title}</h3>
          <p className="text-slate-400 leading-relaxed">{p.desc}</p>
        </motion.div>
      ))}
    </div>
  )
}

function OperatingModes() {
  const plans = [
    {
      name: 'Open Source',
      tagline: 'Self-host the core stack on your metal.',
      features: ['GitHub-first deployments', 'Full source code access', 'Run locally offline', 'Community Slack support'],
      icon: Layers3,
      accent: 'from-sky-500/25 to-transparent'
    },
    {
      name: 'Managed Cloud',
      tagline: 'Pay for execution without infrastructure.',
      features: ['Hosted GPU workloads', 'Queued executions', 'Dashboard history logging', 'Hourly scheduled refreshes'],
      icon: Cloud,
      accent: 'from-emerald-500/25 to-transparent'
    },
    {
      name: 'Enterprise',
      tagline: 'Custom deployments for large organizations.',
      features: ['On-Prem deployments', 'SAML / SSO integrations', 'Unlimited team members', 'Dedicated priority support'],
      icon: LockKeyhole,
      accent: 'from-amber-500/25 to-transparent'
    }
  ]

  return (
    <div id="commercial" className="w-full max-w-6xl mx-auto px-5 pb-32">
      <div className="text-center mb-16">
        <h2 className="text-4xl font-bold tracking-tight text-white mb-4">Choose an operating mode</h2>
        <p className="text-xl text-slate-400">Scale the platform as your organization's AI needs grow.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {plans.map((plan, i) => (
          <motion.div
            key={i}
            className="relative rounded-3xl overflow-hidden border border-white/10 bg-[#0b1018]"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ delay: i * 0.1, duration: 0.5 }}
          >
            <div className={cn("absolute inset-x-0 top-0 h-40 bg-gradient-to-b opacity-40", plan.accent)}></div>
            <div className="p-8 relative z-10">
              <div className="w-12 h-12 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 flex items-center justify-center mb-6 text-white">
                <plan.icon className="w-5 h-5" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">{plan.name}</h3>
              <p className="text-sm border-b border-white/10 pb-6 mb-6 text-slate-400 h-16">{plan.tagline}</p>
              
              <ul className="space-y-4 mb-8">
                {plan.features.map(f => (
                  <li key={f} className="flex items-start text-sm text-slate-300">
                    <CheckCircle2 className="w-5 h-5 text-emerald-400 mr-3 shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

export function LandingPage() {
  return (
    <div className="min-h-screen bg-[#070b12] text-slate-300 font-sans selection:bg-sky-500/30">
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute inset-x-0 top-[-20%] h-[700px] w-full rounded-[100%] bg-sky-500/5 blur-[120px]" />
      </div>

      <header className="fixed top-0 inset-x-0 z-50 border-b border-white/5 bg-[#070b12]/80 backdrop-blur-xl">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-5 py-4 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl font-bold bg-sky-400/10 text-sky-400 border border-sky-400/20">
              <Fish className="w-5 h-5" />
            </div>
            <span className="text-base font-bold text-white tracking-tight">MCP Tuna</span>
          </div>
          
          <nav className="hidden items-center gap-8 text-sm font-medium text-slate-400 md:flex">
            <a href="#commercial" className="hover:text-white transition-colors">Pricing Options</a>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="hover:text-white transition-colors">Documentation</a>
          </nav>

          <div className="flex items-center gap-3">
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
              className={cn(buttonVariants({ variant: 'ghost', size: 'sm' }), "hidden md:inline-flex text-slate-300 hover:text-white hover:bg-white/5")}
            >
              Sign in via GitHub
            </a>
            <Link
              to="/dashboard"
              className={cn(buttonVariants({ size: 'sm' }), 'rounded-lg bg-white text-black hover:bg-slate-200')}
            >
              Open Workspace
            </Link>
          </div>
        </div>
      </header>

      <main className="relative z-10 w-full overflow-hidden pt-10">
        <HeroSection />
        <MockTerminal />
        <ValueProps />
        <OperatingModes />
      </main>

      <footer className="border-t border-white/5 bg-[#04070a] py-12 text-center text-sm text-slate-500">
         <p>© {new Date().getFullYear()} MCP Tuna. Built for modern agents.</p>
      </footer>
    </div>
  )
}
