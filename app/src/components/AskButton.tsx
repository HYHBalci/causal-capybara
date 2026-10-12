import { useResearch } from '../research/store'
import { useStore } from '../store'
import type { ResearchSelection } from '../research/types'

export function AskButton({ selection, query, label = 'Ask about this', disabled = false }: { selection: ResearchSelection; query?: string; label?: string; disabled?: boolean }) {
  return <button className="btn ghost sm ask-button" disabled={disabled} onClick={() => { useStore.getState().setExplain(null); useResearch.getState().ask(selection, query) }} aria-label={`${label}: ${selection.title}`}>{label}</button>
}
