/** The signature interaction.
 *
 * A persistent bar at the top of every analysis, the way a browser has an
 * address bar. Completing this sentence *is* starting the analysis. The slots
 * change with the design card. It is stored as structured fields, never as a
 * blob of English -- the sentence you read is derived from them.
 */

import { useMemo, useState } from 'react'
import { useStore } from '../store'
import type { ColumnMeta } from '../types'

interface Slot {
  role: string
  prefix: string
  placeholder: string
  kind: 'column' | 'columns' | 'text' | 'number' | 'value'
  help: string
  /** How the history list names this slot: "Set the treatment to re78". The
   *  role id — "treated_unit", "event_time" — is a field name, not a name. */
  noun: string
}

const BASE_SLOTS: Slot[] = [
  { role: 'treatment', prefix: 'Effect of', placeholder: 'treatment', kind: 'column',
    noun: 'the treatment',
    help: 'The programme, policy, drug or exposure whose effect you want.' },
  { role: 'outcome', prefix: 'on', placeholder: 'outcome', kind: 'column',
    noun: 'the outcome',
    help: 'What you think it changed.' },
]

const POPULATION_SLOT: Slot = {
  role: 'population', prefix: 'for', placeholder: 'everyone in the data', kind: 'text',
  noun: 'who the answer is about',
  help: 'Who the answer is about. Leave it blank for everyone; type a filter to narrow it.',
}

const DESIGN_SLOTS: Record<string, Slot[]> = {
  rd: [
    { role: 'running', prefix: 'at a cutoff on', placeholder: 'running variable', kind: 'column',
      noun: 'the score that decided eligibility',
      help: 'The score, age or distance that decided who was eligible. Its technical name is the '
          + 'running variable.' },
    { role: 'cutoff', prefix: '=', placeholder: 'cutoff', kind: 'value',
      noun: 'the cutoff',
      help: 'The value at which the programme switched on. Type the number in — there is nothing on '
          + 'the chart to drag.' },
  ],
  iv: [
    { role: 'instruments', prefix: 'using instrument', placeholder: 'instrument', kind: 'columns',
      noun: 'the instrument',
      help: 'Something that pushed people into treatment but has no other route to the outcome. Its '
          + 'technical name is an instrument.' },
  ],
  did: [
    { role: 'unit', prefix: 'across', placeholder: 'unit', kind: 'column',
      noun: 'the unit',
      help: 'The thing that is observed over and over: states, hospitals, schools. One row per unit '
          + 'per period.' },
    { role: 'time', prefix: 'over', placeholder: 'time', kind: 'column',
      noun: 'the time period',
      help: 'The column holding the year, month, wave or date each row belongs to.' },
  ],
  synth: [
    { role: 'treated_unit', prefix: 'for the treated unit', placeholder: 'unit', kind: 'value',
      noun: 'the treated unit',
      help: 'The one place, firm or hospital that got the intervention. Type its name exactly as it '
          + 'is written in the data.' },
  ],
  its: [
    { role: 'event_time', prefix: 'interrupted at', placeholder: 'date', kind: 'value',
      noun: 'the date the policy started',
      help: 'When the policy took effect. Everything before it sets the old pattern; everything '
          + 'after it is what you are testing.' },
  ],
  mediation: [
    { role: 'mediator', prefix: 'through', placeholder: 'mediator', kind: 'columns',
      noun: 'the step in between',
      help: 'The step in between that you think the effect travels through — training raises skills, '
          + 'and skills raise wages. Its technical name is the mediator.' },
  ],
}

const COMPARISON_SLOT: Record<string, Slot> = {
  observational: { role: 'comparison', prefix: 'compared to', placeholder: 'the ones who did not get it',
    kind: 'text', noun: 'the comparison',
    help: 'What you are holding it up against, in your own words. It becomes the first line of the report.' },
  rct: { role: 'comparison', prefix: 'compared to', placeholder: 'the group given nothing', kind: 'text',
    noun: 'the comparison',
    help: 'What you are holding it up against, in your own words — the control group.' },
  did: { role: 'comparison', prefix: 'compared to',
    placeholder: 'units the programme had not reached yet', kind: 'text', noun: 'the comparison',
    help: 'What you are holding it up against, in your own words. Units that get the programme later '
        + 'can stand in for units that got it earlier, while they are still waiting: the '
        + 'not-yet-treated comparison.' },
  synth: { role: 'comparison', prefix: 'compared to', placeholder: 'a blend of the untouched places',
    kind: 'text', noun: 'the comparison',
    help: 'What you are holding it up against, in your own words. Here it is built by blending places '
        + 'the policy never touched until the blend behaves like the treated one did beforehand — a '
        + 'synthetic twin.' },
  its: { role: 'comparison', prefix: 'compared to', placeholder: 'where the old trend was heading',
    kind: 'text', noun: 'the comparison',
    help: 'What you are holding it up against, in your own words. Here it is the line the outcome was '
        + 'already following before the policy, carried forward.' },
}

export function QuestionStrip() {
  const spec = useStore((s) => s.spec)
  const columns = useStore((s) => s.columns)
  const savingSpec = useStore((s) => s.savingSpec)
  const setExplain = useStore((s) => s.setExplain)
  const newSpec = useStore((s) => s.newSpec)

  const slots = useMemo(() => {
    if (!spec) return BASE_SLOTS
    const design = spec.design
    const extra = DESIGN_SLOTS[design] ?? []
    const cmp = COMPARISON_SLOT[design]
    if (design === 'rd') return [BASE_SLOTS[1], ...extra, POPULATION_SLOT]
    if (design === 'its') return [BASE_SLOTS[1], ...extra, POPULATION_SLOT]
    return [...BASE_SLOTS, ...extra, POPULATION_SLOT, ...(cmp ? [cmp] : [])]
  }, [spec?.design])

  // An empty bar with nothing to press taught nobody what the bar was for. It
  // is the first thing on every screen, so it says what it holds and offers the
  // one action that fills it.
  if (!spec) {
    return (
      <div className="qstrip empty row" style={{ gap: 8 }}>
        <span className="hint grow">
          This bar holds your question — the effect of what, on what, for whom. Filling it in is how an
          analysis starts.
        </span>
        <button className="btn primary sm" onClick={() => void newSpec()}>Start a question</button>
      </div>
    )
  }

  return (
    <div className="qstrip" role="group" aria-label="Causal question">
      <div className="qstrip-line">
        {slots.map((slot) => (
          <QuestionSlot key={slot.role} slot={slot} columns={columns} />
        ))}
      </div>
      <div className="qstrip-meta">
        {savingSpec && <span className="spinner" aria-label="Saving" />}
        {spec.estimand && (
          <button
            className="chip teal"
            title={`Read what “${spec.estimand}” means — which people this number averages over`}
            onClick={() => setExplain(`estimand.${spec.estimand!.toLowerCase()}`)}
          >
            {spec.estimand}
          </button>
        )}
      </div>
    </div>
  )
}

function QuestionSlot({ slot, columns }: { slot: Slot; columns: ColumnMeta[] }) {
  const spec = useStore((s) => s.spec)!
  const updateSpec = useStore((s) => s.updateSpec)
  const setRole = useStore((s) => s.setRole)
  const setView = useStore((s) => s.setView)
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const [over, setOver] = useState(false)

  const isQuestionField = slot.role === 'population' || slot.role === 'comparison'
  const raw = isQuestionField
    ? (spec.question as Record<string, unknown>)[slot.role]
    : (spec.roles as Record<string, unknown>)[slot.role]
  const value = Array.isArray(raw) ? (raw as string[]).join(', ') : (raw as string | number | null)
  const filled = value !== null && value !== undefined && value !== ''

  // The history list is read by whoever filled this bar in, so it names the slot
  // the way the bar does rather than by its role id.
  const said = (v: unknown) => {
    const empty = v === null || v === undefined || v === '' || (Array.isArray(v) && !v.length)
    return empty ? `Cleared ${slot.noun}` : `Set ${slot.noun} to ${Array.isArray(v) ? v.join(', ') : v}`
  }

  const commit = (v: unknown) => {
    if (isQuestionField) {
      updateSpec((s) => ({ ...s, question: { ...s.question, [slot.role]: v as string } }),
        said(v), { [slot.role]: v })
    } else if (slot.role === 'treatment' || slot.role === 'outcome') {
      // These two live in both places: the sentence and the roles.
      updateSpec(
        (s) => ({
          ...s,
          question: { ...s.question, [slot.role]: v as string },
          roles: { ...s.roles, [slot.role]: v as string },
        }),
        said(v),
        { role: slot.role, value: v },
      )
    } else {
      setRole(slot.role, v, said(v))
    }
    setOpen(false)
    setFilter('')
  }

  if (slot.kind === 'text') {
    // Size the box to its words. A fixed-width input showed "people eligible
    // for the trainin" and hid the rest; the sentence is the point of the strip.
    const text = (value as string) ?? ''
    return (
      <span className="qslot-wrap">
        <span className="qslot-prefix">{slot.prefix}</span>
        <input
          className={`qslot text${filled ? ' filled' : ''}`}
          size={Math.max(12, Math.min(48, (text || slot.placeholder).length + 1))}
          value={text}
          placeholder={slot.placeholder}
          title={slot.help}
          aria-label={`${slot.prefix} ${slot.placeholder}`}
          onChange={(e) => commit(e.target.value || null)}
        />
      </span>
    )
  }

  if (slot.kind === 'value' || slot.kind === 'number') {
    return (
      <span className="qslot-wrap">
        <span className="qslot-prefix">{slot.prefix}</span>
        <input
          className={`qslot value${filled ? ' filled' : ''}`}
          value={value === null || value === undefined ? '' : String(value)}
          placeholder={slot.placeholder}
          title={slot.help}
          aria-label={`${slot.prefix} ${slot.placeholder}`}
          onChange={(e) => {
            const t = e.target.value
            const n = Number(t)
            commit(t === '' ? null : Number.isFinite(n) && t.trim() !== '' ? n : t)
          }}
        />
      </span>
    )
  }

  const multiple = slot.kind === 'columns'
  const selected = multiple ? ((raw as string[]) ?? []) : []
  const options = columns.filter(
    (c) => !filter || c.name.toLowerCase().includes(filter.toLowerCase()),
  )

  return (
    <span className="qslot-wrap">
      <span className="qslot-prefix">{slot.prefix}</span>
      <span
        className={`qslot col${filled ? ' filled' : ''}${over ? ' over' : ''}`}
        onDragOver={(e) => {
          if (e.dataTransfer.types.includes('text/capy-variable')) {
            e.preventDefault()
            setOver(true)
          }
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          const name = e.dataTransfer.getData('text/capy-variable')
          if (!name) return
          commit(multiple ? [...new Set([...selected, name])] : name)
        }}
      >
        <button
          className="qslot-btn"
          onClick={() => setOpen((o) => !o)}
          title={slot.help}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-label={`${slot.prefix} ${slot.placeholder}`}
        >
          {filled ? String(value) : slot.placeholder}
          <span className="qslot-caret" aria-hidden>▾</span>
        </button>
        {open && (
          <div className="qslot-menu card" role="listbox">
            <input
              autoFocus
              className="qslot-filter"
              placeholder="Filter variables"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setOpen(false)
                if (e.key === 'Enter' && options[0]) {
                  commit(multiple ? [...new Set([...selected, options[0].name])] : options[0].name)
                }
              }}
            />
            <div className="qslot-options scroll">
              {!columns.length && (
                <div className="pad">
                  <p className="hint tiny" style={{ margin: '0 0 6px' }}>
                    There is no data open yet, so there are no columns to choose from.
                  </p>
                  <button className="btn sm" onClick={() => setView('data')}>Open a data file</button>
                </div>
              )}
              {options.map((c) => (
                <button
                  key={c.name}
                  className={`qslot-option${selected.includes(c.name) ? ' on' : ''}`}
                  role="option"
                  aria-selected={selected.includes(c.name)}
                  onClick={() =>
                    commit(
                      multiple
                        ? selected.includes(c.name)
                          ? selected.filter((s) => s !== c.name)
                          : [...selected, c.name]
                        : c.name,
                    )
                  }
                >
                  <span className={`kind-icon ${c.kind}`} aria-hidden />
                  <span className="grow">{c.name}</span>
                  <span className="tiny hint">{c.kind}</span>
                </button>
              ))}
              {columns.length > 0 && !options.length && (
                <div className="hint tiny pad">No column here is called “{filter}”.</div>
              )}
            </div>
            {filled && (
              <button className="btn ghost sm" onClick={() => commit(multiple ? [] : null)}>
                Clear
              </button>
            )}
          </div>
        )}
      </span>
    </span>
  )
}
