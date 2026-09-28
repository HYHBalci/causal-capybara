import { useStore } from '../store'

export function WorkspaceBar() {
  const view = useStore((state) => state.view)
  const setView = useStore((state) => state.setView)
  const project = useStore((state) => state.project)
  const spec = useStore((state) => state.spec)
  const saving = useStore((state) => state.savingSpec)
  const saveError = useStore((state) => state.saveError)
  const flush = useStore((state) => state.flushSpec)
  return <nav className="workspace-bar" aria-label="Workspace">
    <div className="workspace-links">
      <button aria-current={view === 'welcome' ? 'page' : undefined} onClick={() => setView('welcome')}>Home</button>
      {project && <button aria-current={!['welcome', 'learn', 'literature', 'settings', 'engines'].includes(view) ? 'page' : undefined} onClick={() => setView(project.has_data ? 'board' : 'data')}>Workspace</button>}
      <button aria-current={view === 'learn' ? 'page' : undefined} onClick={() => setView('learn')}>Method catalogue</button>
      <button aria-current={view === 'literature' ? 'page' : undefined} onClick={() => setView('literature')}>Literature</button>
    </div>
    <div className="workspace-save" role="status">
      {saveError ? <><span className="warn">Question edits not saved</span><button className="btn sm" onClick={() => void flush().catch(() => {})}>Retry save</button></>
        : project ? <span>{saving ? 'Saving question…' : spec ? 'Question saved locally' : 'Project stored locally'}</span> : <span>Your data stays on this computer</span>}
    </div>
  </nav>
}
