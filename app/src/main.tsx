import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { ErrorBoundary } from './ErrorBoundary'

// Self-hosted, bundled with the app. The theme has always named Inter, Source
// Serif and JetBrains Mono; until now none of them was ever loaded, so every
// screen silently fell back to Segoe UI and the type had no hierarchy at all.
// Local files rather than a CDN: this is an offline-first desktop studio.
import '@fontsource-variable/inter'
import '@fontsource/source-serif-4/400.css'
import '@fontsource/source-serif-4/600.css'
import '@fontsource-variable/jetbrains-mono'

import './theme.css'
import './app.css'
import './panels.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
