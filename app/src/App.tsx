import { useCallback, useEffect, useRef } from 'react'
import { CommandPalette, GuidedMiniMap, JobBar, MenuBar, StatusBar, Toasts } from './components/Chrome'
import { ExplainPane, Inspector } from './components/Inspector'
import { WorkspaceBar } from './components/WorkspaceBar'
import { useDialogFocus } from './components/useDialogFocus'
import { Literature } from './canvas/Literature'
import { ProjectNavigator } from './components/ProjectNavigator'
import { QuestionStrip } from './components/QuestionStrip'
import { Compare } from './canvas/Compare'
import { Dashboard } from './canvas/Dashboard'
import { DagEditor } from './canvas/DagEditor'
import { DataSheet } from './canvas/DataSheet'
import { DesignBoard } from './canvas/DesignBoard'
import { DesignCards } from './canvas/DesignCards'
import { Diagnose } from './canvas/Diagnose'
import { EngineDown } from './canvas/EngineDown'
import { Learn } from './canvas/Learn'
import { CodeView, Engines, ReportView, SimLab } from './canvas/Panels'
import { Recommend } from './canvas/Recommend'
import { Settings } from './canvas/Settings'
import { Welcome } from './canvas/Welcome'
import { useStore } from './store'

/** Screens that are part of the window itself and need nothing computed.
 *
 * Everything else asks the engine a question, so while the engine is down they
 * are replaced by the screen that explains how to get one -- rather than by a
 * spinner, an empty panel, or the error toast the request would otherwise
 * produce. */
const WORKS_WITHOUT_ENGINE = new Set(['learn', 'literature', 'settings', 'engines'])

/** Screens that are about the application rather than about a project.
 *
 * The navigator and the inspector are both views onto the open project, and on
 * these three they have nothing to say -- they were rendering "No project open"
 * and "Select a variable and its details appear here" beside a page that has
 * neither, spending over 500px of a 1440px window on two apologies. On a 1280px
 * laptop that left the catalogue article too narrow to read comfortably. */
const FULL_WIDTH_VIEWS = new Set(['welcome', 'learn', 'literature', 'settings', 'engines'])

export default function App() {
  const booted = useStore((s) => s.booted)
  const boot = useStore((s) => s.boot)
  const view = useStore((s) => s.view)
  const project = useStore((s) => s.project)
  const engineDown = useStore((s) => s.engineDown)
  const mode = useStore((s) => s.mode)
  const setPalette = useStore((s) => s.setPalette)
  const setTheme = useStore((s) => s.setTheme)
  const theme = useStore((s) => s.theme)
  const newSpec = useStore((s) => s.newSpec)
  const goBack = useStore((s) => s.goBack)
  const goForward = useStore((s) => s.goForward)

  useEffect(() => { void boot() }, [])
  useEffect(() => { setTheme(theme) }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey
      if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); setPalette(true) }
      if (mod && e.key.toLowerCase() === 'q') { e.preventDefault(); void newSpec() }
      if (e.altKey && e.key === 'ArrowLeft') { e.preventDefault(); goBack() }
      if (e.altKey && e.key === 'ArrowRight') { e.preventDefault(); goForward() }
      if (e.key === 'Escape') { setPalette(false); useStore.getState().setExplain(null) }
    }
    // The side buttons on a mouse mean back and forward everywhere else.
    const onMouse = (e: MouseEvent) => {
      if (e.button === 3) { e.preventDefault(); goBack() }
      if (e.button === 4) { e.preventDefault(); goForward() }
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('mouseup', onMouse)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('mouseup', onMouse)
    }
  }, [])

  // The only thing worth waiting for is the first answer. Everything after it
  // renders the real window, engine or no engine.
  if (!booted) {
    return (
      <div className="boot">
        <div className="boot-card card">
          <h1>Causal Capybara</h1>
          <p className="welcome-tag">Clear questions. Credible evidence.</p>
          <p className="row" style={{ gap: 8, marginTop: 14 }}>
            <span className="spinner" /> Opening…
          </p>
        </div>
      </div>
    )
  }

  const wide = FULL_WIDTH_VIEWS.has(view) || (engineDown && !WORKS_WITHOUT_ENGINE.has(view))
  return (
    <div className="app">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <MenuBar />
      <WorkspaceBar />
      <div className={`app-body${wide ? ' full-width' : ''}`}>
        {!wide && <ProjectNavigator />}
        <main id="main-content" tabIndex={-1} className="canvas col grow">
          {project && !wide && <QuestionStrip />}
          {project && !wide && mode === 'guided' && view !== 'welcome' && <GuidedMiniMap />}
          <div className="canvas-body col grow">
            <CanvasView view={view} />
          </div>
        </main>
        {!wide && <Inspector />}
      </div>
      <JobBar />
      <StatusBar />
      <CommandPalette />
      <Toasts />
      {wide && <ExplanationDialog />}
    </div>
  )
}

function CanvasView({ view }: { view: string }) {
  const project = useStore((s) => s.project)
  const engineDown = useStore((s) => s.engineDown)
  if (engineDown && !WORKS_WITHOUT_ENGINE.has(view)) return <EngineDown />
  switch (view) {
    case 'learn': return <Learn />
    case 'literature': return <Literature />
    case 'settings': return <Settings />
    case 'engines': return <Engines />
    default: break
  }
  if (!project) return <Welcome />
  switch (view) {
    case 'welcome': return <Welcome />
    case 'data': return <DataSheet />
    case 'design': return <DesignCards />
    case 'board': return <DesignBoard />
    case 'diagnose': return <Diagnose />
    case 'recommend': return <Recommend />
    case 'dashboard': return <Dashboard />
    case 'compare': return <Compare />
    case 'sim': return <SimLab />
    case 'dag': return <DagEditor />
    case 'report': return <ReportView />
    case 'code': return <CodeView />
    default: return <DataSheet />
  }
}

function ExplanationDialog() {
  const key = useStore((state) => state.explainKey)
  const setExplain = useStore((state) => state.setExplain)
  const ref = useRef<HTMLDivElement>(null)
  const close = useCallback(() => setExplain(null), [setExplain])
  useDialogFocus(ref, !!key, close)
  if (!key) return null
  return <div className="palette-scrim" onClick={close}>
    <div className="explanation-dialog card scroll" ref={ref} role="dialog" aria-modal="true" aria-label="Explanation" onClick={(event) => event.stopPropagation()}>
      <button className="btn ghost explanation-close" onClick={close}>Close explanation ✕</button>
      <ExplainPane />
    </div>
  </div>
}
