import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The desktop shell renders with web tech; the user launches Causal Capybara,
// not a browser. In development this is a plain Vite server talking to the
// sidecar on 127.0.0.1:8760.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    // Vite walks the whole project to set up file watches. src-tauri/target is
    // a Rust build directory -- hundreds of thousands of files, rewritten while
    // cargo runs -- and watching it kills the dev server outright with
    // "EBUSY: resource busy or locked" the moment a build touches a binary it
    // has open. Nothing in there is a source file.
    watch: {
      ignored: ['**/src-tauri/target/**', '**/src-tauri/gen/**'],
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8760',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
  build: { outDir: 'dist', sourcemap: true, target: 'es2022' },
})
