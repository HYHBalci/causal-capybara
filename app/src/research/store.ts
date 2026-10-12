import { create } from 'zustand'
import type { ResearchSelection } from './types'

interface ResearchState {
  assistantOpen: boolean
  selection: ResearchSelection | null
  query: string
  treeSelection: string | null
  ask: (selection?: ResearchSelection, query?: string) => void
  close: () => void
  selectTree: (id: string | null) => void
}
export const useResearch = create<ResearchState>((set) => ({
  assistantOpen: false, selection: null, query: '', treeSelection: null,
  ask: (selection, query) => set({ assistantOpen: true, selection: selection ?? null, query: query ?? (selection ? `Explain ${selection.title}, its assumptions, limitations and sources.` : '') }),
  close: () => set({ assistantOpen: false }),
  selectTree: (id) => set({ treeSelection: id }),
}))
