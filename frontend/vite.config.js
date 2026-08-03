import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5175,
    proxy: {
      '/api/python': {
        target: process.env.VITE_PYTHON_API_URL || 'http://localhost:8100',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/python/, '')
      },
      '/api/java': {
        target: process.env.VITE_JAVA_API_URL || 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/java/, '')
      }
    }
  }
})
