import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5175,
    proxy: {
      // 本地开发：后端 uvicorn 默认 8000（docker 模式端口 8100 通过
      // VITE_PYTHON_API_URL 覆盖，例如 http://localhost:8100）
      '/api': {
        target: process.env.VITE_PYTHON_API_URL || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
