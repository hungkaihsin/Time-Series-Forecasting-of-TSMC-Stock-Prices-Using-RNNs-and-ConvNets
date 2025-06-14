import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // your frontend port
    proxy: {
      '/api': {
        target: 'http://localhost:5001', // Flask backend
        changeOrigin: true,
      }
    }
  }
})