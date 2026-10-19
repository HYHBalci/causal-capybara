import { create } from 'zustand'
import type { KnowledgeCitation, ResearchRecord } from './types'

export interface ChatTurn {
  id: string
  question: string
  status: 'pending' | 'completed' | 'failed' | 'cancelled'
  provider: string
  answer?: string
  reply?: string
  filename?: string
  error?: string
  citations?: KnowledgeCitation[]
  warning?: string
  changes?: ResearchRecord
  // A revision is a comparison value, never an approval or live context token.
  proposalContext?: { project_id: string; spec_id: string; base_revision: string }
}
export interface Conversation { draft: string; turns: ChatTurn[] }
const EMPTY: Conversation = { draft: '', turns: [] }
export const CHAT_STORAGE_KEY = 'capy.assistant.conversations.v1'

/** Read only our versioned display data; never restore credentials or approval state. */
export function readConversations(raw: string | null): Record<string, Conversation> {
  if (!raw) return {}
  const parsed = JSON.parse(raw)
  if (parsed.version !== 1 || !Array.isArray(parsed.conversations)) throw new Error('Unknown conversation format')
  const result: Record<string, Conversation> = {}
  for (const [scope, chat] of parsed.conversations) {
    if (typeof scope !== 'string' || !chat || typeof chat.draft !== 'string' || !Array.isArray(chat.turns)) throw new Error('Invalid conversation')
    const turns = chat.turns.map((turn: ChatTurn): ChatTurn => {
      if (!turn || typeof turn.id !== 'string' || typeof turn.question !== 'string' || typeof turn.provider !== 'string'
          || !['pending', 'completed', 'failed', 'cancelled'].includes(turn.status)
          || (turn.answer !== undefined && typeof turn.answer !== 'string')) throw new Error('Invalid message')
      return { id: turn.id, question: turn.question, provider: turn.provider,
        status: turn.status === 'pending' ? 'cancelled' : turn.status,
        answer: turn.answer, reply: typeof turn.reply === 'string' ? turn.reply : undefined,
        filename: typeof turn.filename === 'string' ? turn.filename : undefined,
        error: turn.status === 'pending' ? 'The app closed before this reply finished. Try the question again.' : typeof turn.error === 'string' ? turn.error : undefined,
        citations: Array.isArray(turn.citations) ? turn.citations.filter((c) => c && typeof c.id === 'string') : undefined,
        warning: typeof turn.warning === 'string' ? turn.warning : undefined,
        changes: turn.changes && typeof turn.changes === 'object' && !Array.isArray(turn.changes) ? turn.changes : undefined,
        proposalContext: turn.proposalContext && ['project_id', 'spec_id', 'base_revision'].every((key) => typeof (turn.proposalContext as ResearchRecord)[key] === 'string')
          ? { project_id: turn.proposalContext.project_id, spec_id: turn.proposalContext.spec_id, base_revision: turn.proposalContext.base_revision } : undefined }
    })
    Object.defineProperty(result, scope, { value: { draft: chat.draft.slice(0, 4000), turns }, enumerable: true, configurable: true, writable: true })
  }
  return result
}

/** Bounded earlier dialogue, separate from the current question and study snapshot. */
export function conversationContext(turns: ChatTurn[]) {
  const complete = turns.filter((turn) => turn.status === 'completed' && turn.answer)
  let shortened = false
  const encoder = new TextEncoder()
  const clip = (text: string) => {
    // Strip suggestion transport JSON; a past proposal is not an applied change.
    const plain = text.replace(/```capy-proposal\s*\n[\s\S]*?\n```/g, '').trim()
    let kept = '', bytes = 0
    for (const char of plain) {
      // Budget the serialized text too: code and control characters need escaping.
      bytes += encoder.encode(JSON.stringify(char).slice(1, -1)).length
      if (bytes > 1700) { shortened = true; return kept + '\n[Earlier message shortened for context.]' }
      kept += char
    }
    return plain || '[No prose in this earlier reply.]'
  }
  return { messages: complete.slice(-4).flatMap((turn) => [
    { role: 'user' as const, content: clip(turn.question) }, { role: 'assistant' as const, content: clip(turn.reply ?? turn.answer!) },
  ]), omitted: Math.max(0, complete.length - 4), shortened }
}

interface ConversationsState {
  conversations: Record<string, Conversation>
  storageError: string
  update: (scope: string, change: (chat: Conversation) => Conversation) => void
  clear: (scope: string) => void
}
let initial: Record<string, Conversation> = {}, storageError = ''
try { initial = readConversations(localStorage.getItem(CHAT_STORAGE_KEY)) }
catch { storageError = 'Saved chats could not be read. New messages will stay in this session until local saving works.' }
function persist(conversations: Record<string, Conversation>) {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify({ version: 1, conversations: Object.entries(conversations) }))
    return ''
  } catch { return 'Local saving is unavailable or storage is full. This chat is still open; save important answers as Markdown before closing the app.' }
}
export const useConversations = create<ConversationsState>((set) => ({
  conversations: initial, storageError,
  update: (scope, change) => set((state) => {
    const conversations = { ...state.conversations, [scope]: change(state.conversations[scope] ?? EMPTY) }
    return { conversations, storageError: persist(conversations) }
  }),
  clear: (scope) => set((state) => {
    const conversations = { ...state.conversations }; delete conversations[scope]
    return { conversations, storageError: persist(conversations) }
  }),
}))
export const emptyConversation = EMPTY
