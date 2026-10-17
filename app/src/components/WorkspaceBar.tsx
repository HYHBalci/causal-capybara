import { useStore } from '../store'
import { useResearch } from '../research/store'

export function WorkspaceBar() {
  const view = useStore((state) => state.view)
  const navigate = useStore((state) => state.setView)
  const setView: typeof navigate = (next) => { useResearch.getState().setExpanded(false); navigate(next) }
  const assistantOpen = useResearch((state) => state.assistantOpen)
  const project = useStore((state) => state.project)
  const spec = useStore((state) => state.spec)
  const saving = useStore((state) => state.savingSpec)
  const saveError = useStore((state) => state.saveError)
  const flush = useStore((state) => state.flushSpec)
  return <nav className="workspace-bar" aria-label="Workspace">
    <div className="workspace-links">
      <button aria-current={view === 'welcome' ? 'page' : undefined} onClick={() => setView('welcome')}>Home</button>
      {project && <button aria-current={!['welcome', 'learn', 'literature', 'research', 'settings', 'engines'].includes(view) ? 'page' : undefined} onClick={() => setView(project.has_data ? 'board' : 'data')}>Workspace</button>}
      <button aria-current={view === 'learn' ? 'page' : undefined} onClick={() => setView('learn')}>Method catalogue</button>
      <button aria-current={view === 'literature' ? 'page' : undefined} onClick={() => setView('literature')}>Literature</button>
      <button aria-current={view === 'research' ? 'page' : undefined} onClick={() => setView('research')}>Research tree</button>
    </div>
    <div className="workspace-save" role="status">
      {saveError ? <><span className="warn">Question edits not saved</span><button className="btn sm" onClick={() => void flush().catch(() => {})}>Retry save</button></>
        : project ? <span>{saving ? 'Saving question…' : spec ? 'Question saved locally' : 'Project stored locally'}</span> : <span>Your data stays on this computer</span>}
    </div>
    <button className="btn primary assistant-launch" aria-expanded={assistantOpen} aria-controls="research-assistant-panel" title="Research assistant (Ctrl+Shift+A)" onClick={() => { if (assistantOpen) useResearch.getState().close(); else useResearch.getState().ask() }}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-6 4V6a2 2 0 0 1 2-2Z"/><path d="M7 9h10M7 13h7"/></svg>
      Research assistant
    </button>
  </nav>
}
