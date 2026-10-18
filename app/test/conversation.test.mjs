import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

const stored = new Map()
globalThis.localStorage = { getItem: (key) => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value) }
const built = await build({ stdin: { contents: "export * from './src/research/conversation'", resolveDir: process.cwd(), loader: 'ts' }, bundle: true, write: false, format: 'esm', platform: 'node' })
const { readConversations, conversationContext, useConversations, CHAT_STORAGE_KEY } = await import('data:text/javascript;base64,' + Buffer.from(built.outputFiles[0].text).toString('base64'))
const turn = (id, status = 'completed') => ({ id, question: `Question ${id}`, answer: `Answer ${id}`, status, provider: 'Codex' })

test('follow-ups include only completed pairs, newest last, within the UTF-8 budget', () => {
  const turns = Array.from({ length: 8 }, (_, i) => turn(String(i)))
  turns.push(turn('pending', 'pending'), turn('failed', 'failed'), turn('stopped', 'cancelled'))
  const history = conversationContext(turns)
  assert.equal(history.omitted, 4)
  assert.deepEqual(history.messages.map((m) => m.content), ['Question 4', 'Answer 4', 'Question 5', 'Answer 5', 'Question 6', 'Answer 6', 'Question 7', 'Answer 7'])
  const huge = conversationContext(turns.map((t) => ({ ...t, answer: '💬'.repeat(4000) })))
  assert(huge.shortened)
  assert(Buffer.byteLength(JSON.stringify(huge.messages)) < 16384)
  assert(!JSON.stringify(huge.messages).includes('\ufffd'))
  const escaped = conversationContext(turns.map((t) => ({ ...t, question: '\\"\n'.repeat(1000), answer: '\\"\n'.repeat(8000) })))
  assert(Buffer.byteLength(JSON.stringify(escaped.messages)) < 16384)
  assert(!JSON.stringify(conversationContext([{ ...turn('proposal'), answer: 'Suggestion\n```capy-proposal\n{"title":"not applied"}\n```' }])).includes('not applied'))
  assert.equal(conversationContext([{ ...turn('export'), answer: '# Complete export with metadata', reply: 'Actual conversational reply' }]).messages[1].content, 'Actual conversational reply')
})

test('local chats and drafts survive restart, remain scoped and never resume pending jobs', () => {
  useConversations.getState().update('study-A', () => ({ draft: 'Follow up later', turns: [turn('one'), turn('two', 'pending')] }))
  useConversations.getState().update('study-B', () => ({ draft: '', turns: [turn('other')] }))
  const restored = readConversations(stored.get(CHAT_STORAGE_KEY))
  assert.equal(restored['study-A'].draft, 'Follow up later')
  assert.equal(restored['study-A'].turns[1].status, 'cancelled')
  assert.equal(restored['study-B'].turns[0].id, 'other')
  useConversations.getState().clear('study-A')
  assert(!readConversations(stored.get(CHAT_STORAGE_KEY))['study-A'])
  assert(readConversations(stored.get(CHAT_STORAGE_KEY))['study-B'])
})

test('stored display data cannot restore credentials, context tokens or approvals', () => {
  const restored = readConversations(JSON.stringify({ version: 1, conversations: [['study', { draft: '', api_key: 'SECRET', turns: [{ ...turn('one'), approval_token: 'SECRET', context_id: 'SECRET', proposalContext: { project_id: 'p', spec_id: 's', base_revision: 'hash', approval_token: 'SECRET' } }] }]] }))
  assert(!JSON.stringify(restored).includes('SECRET'))
  assert.throws(() => readConversations('{broken'))
  assert.throws(() => readConversations(JSON.stringify({ version: 2, conversations: [] })))
})

test('storage failure keeps the current chat in memory and reports that saving failed', () => {
  localStorage.setItem = () => { throw Error('quota exceeded') }
  useConversations.getState().update('quota', () => ({ draft: 'Do not lose this', turns: [turn('retained')] }))
  assert.equal(useConversations.getState().conversations.quota.draft, 'Do not lose this')
  assert.match(useConversations.getState().storageError, /storage is full/)
})
