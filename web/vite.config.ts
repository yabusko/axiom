import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // B8: the GitHub Pages demo build (.github/workflows/pages.yml, VITE_DEMO=1)
  // is a project page served at /axiom/, not the domain root -- every other
  // build (dev, a real API-backed deploy) keeps the default root base.
  base: process.env.VITE_DEMO === '1' ? '/axiom/' : '/',
})
