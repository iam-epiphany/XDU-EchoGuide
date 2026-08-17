import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5175,
    proxy: {
      // 本地开发：后端 uvicorn 默认 8100（与 .env.example 一致；后端换端口时
      // 可用 VITE_PYTHON_API_URL 覆盖，例如 http://localhost:8000）
      '/api': {
        target: process.env.VITE_PYTHON_API_URL || 'http://localhost:8100',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
