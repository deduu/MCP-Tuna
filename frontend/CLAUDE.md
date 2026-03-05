# Frontend — React Dashboard Rules

## This Directory
React dashboard for MCP Tuna. Communicates with the backend exclusively via MCP gateway
(JSON-RPC 2.0) and OpenAI-compatible chat API. Zero direct pipeline imports.

## Tech Stack
- React 19, TypeScript 5.9, Vite 7.3
- Zustand 5.0 (state management with localStorage persistence)
- TanStack React Query 5.x (server state / data fetching)
- React Router 7.x (routing)
- Tailwind CSS 4.x + CVA (component variants)
- lucide-react (icons), sonner (toasts), cmdk (command palette)
- Framer Motion (animations), react-markdown (rendering)

## Structure
```
frontend/src/
├── api/
│   ├── client.ts              # mcpCall<T>(tool, args) — JSON-RPC 2.0 to /mcp
│   ├── chat-client.ts         # OpenAI-compatible streaming chat client
│   ├── types.ts               # Shared API response types
│   └── hooks/                 # TanStack Query hooks per domain
│       ├── useDatasets.ts
│       ├── useTraining.ts
│       ├── useDeployments.ts
│       ├── useEvaluation.ts
│       ├── usePipeline.ts
│       ├── useSystemResources.ts
│       ├── useToolRegistry.ts
│       └── useToolExecution.ts
├── stores/
│   ├── app.ts                 # UI state (sidebar, command palette) — persisted
│   ├── chat.ts                # Chat messages, streaming, events
│   └── orchestration.ts       # Pipeline step tracking
├── components/
│   ├── ui/                    # Base primitives (button, card, dialog, tabs, badge, etc.)
│   ├── layout/                # AppShell, Header, Sidebar, CommandPalette
│   ├── chat/                  # Chat interface
│   ├── dashboard/             # Dashboard + QuickActions
│   ├── datasets/              # Dataset management
│   ├── training/              # Training job management
│   ├── deployments/           # Model deployment UI
│   ├── evaluation/            # Evaluation workflows
│   ├── pipeline/              # Orchestration workflows
│   ├── settings/              # App configuration
│   └── tools/                 # Tool registry explorer
└── lib/
    └── utils.ts               # cn() — clsx + tailwind-merge
```

## Key Patterns

### API Client (`api/client.ts`)
- `mcpCall<T>(toolName, args)` — posts JSON-RPC 2.0 to `/mcp`, returns typed result
- `mcpListTools()` — lists available MCP tools
- All backend communication goes through these two functions

### Hooks (`api/hooks/`)
- Use `useQuery()` for reads with appropriate `staleTime` and `refetchInterval`
- Use `useMutation()` for writes with `useQueryClient().invalidateQueries()` on success
- Naming: `use<Resource>` for queries, `use<Action><Resource>` for mutations
- All hooks call `mcpCall()` — never fetch raw endpoints directly

### Stores (`stores/`)
- Zustand with `create()` — keep stores small and focused
- Persist UI preferences to localStorage under `mcp-tuna-*` keys
- Don't persist transient state (streaming, orchestration progress)

### Components
- Base UI in `components/ui/` — shadcn-style with `forwardRef` + `cn()` + CVA variants
- Page components use Tabs for section layout, Cards for content blocks
- Loading: Skeleton components; Errors: sonner toasts
- Icons: lucide-react exclusively

## Vite Dev Server
- Port 5173
- Proxy `/v1` → `http://127.0.0.1:8000` (FastAPI backend)
- Proxy `/mcp` → `http://127.0.0.1:8002` (MCP gateway)
- Path alias `@` → `./src`

## Routes
```
/              Dashboard
/tools         Tool Explorer
/chat          Agent Chat
/pipeline      Pipeline Orchestration
/datasets      Dataset Management
/training      Training Jobs
/deployments   Model Deployments
/evaluation    Evaluation & Benchmarking
/settings      Configuration
```

## Rules
- All backend calls go through `mcpCall()` — never bypass the MCP gateway
- New pages must be added to `App.tsx` routes and `QuickActions.tsx` navigation
- New hooks go in `api/hooks/` and follow the TanStack Query pattern
- UI primitives go in `components/ui/` — page-specific components in their domain folder
- No inline styles — use Tailwind classes only
- Toast notifications via `sonner` — no `alert()` or `console.log()` for user feedback
