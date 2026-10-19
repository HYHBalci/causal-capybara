import { create } from 'zustand'
import type { AssistantHandoffClient, ResearchSelection } from './types'

interface ResearchState {
  cli: { client: AssistantHandoffClient; model: string; effort: string }
  setCli: (cli: ResearchState['cli']) => void
  assistantOpen: boolean
  assistantExpanded: boolean
  setExpanded: (expanded: boolean) => void
  selection: ResearchSelection | null
  query: string
  treeSelection: string | null
  ask: (selection?: ResearchSelection, query?: string) => void
  close: () => void
  selectTree: (id: string | null) => void
}
export const useResearch = create<ResearchState>((set) => ({
  cli: { client: 'codex', model: '', effort: '' },
  setCli: (cli) => set({ cli }),
  assistantOpen: false, assistantExpanded: false,
  setExpanded: (assistantExpanded) => set({ assistantExpanded }),
  selection: null, query: '', treeSelection: null,
  ask: (selection, query) => set({ assistantOpen: true, ...(selection || query !== undefined ? { selection: selection ?? null, query: query ?? `Explain ${selection!.title}, its assumptions, limitations and sources.` } : {}) }),
  close: () => set({ assistantOpen: false, assistantExpanded: false, query: '', selection: null }),
  selectTree: (id) => set({ treeSelection: id }),
}))
