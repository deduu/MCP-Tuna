import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

const apiTarget = process.env.VITE_PROXY_API_TARGET ?? 'http://127.0.0.1:8000'
const mcpTarget = process.env.VITE_PROXY_MCP_TARGET ?? 'http://127.0.0.1:8002'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/v1': apiTarget,
      '/mcp': mcpTarget,
    },
  },
})
