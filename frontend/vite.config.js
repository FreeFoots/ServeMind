import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Frontend-specific variables live in frontend/.env.
// Only VITE_* values are exposed to browser code by Vite.
export default defineConfig({ plugins: [vue()] })
