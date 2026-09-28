/** One way of showing an assumption, used everywhere one appears.
 *
 * The inspector, the Diagnose page and the ledger each drew these differently:
 * one was clickable, two were not, and one showed raw catalogue ids with the
 * underscores swapped for spaces. Someone who learned in one place that an
 * assumption opens an explanation found it inert in the next. Now every
 * assumption is a button that opens its entry, every status carries its
 * meaning, and the wording comes from the catalogue rather than the id.
 */

import { useStore } from '../store'
import type { Assumption, AssumptionStatus } from '../types'

/** The five words a status may say, and what each one actually means.
 *
 * The vocabulary is deliberately short of "passed". A diagnostic can fail to
 * contradict an assumption; it cannot show one to be true. But a beginner
 * reading "not contradicted" in a table cell has no way to tell whether that is
 * good news, so every status carries a plain sentence with it.
 */
export const ASSUMPTION_STATUS: Record<string, { word: string; means: string }> = {
  assumed: {
    word: 'Assumed',
    means: 'Taken on faith. No diagnostic in this design can check it, so it rests on your argument.',
  },
  supported: {
    word: 'Not contradicted',
    means: 'A diagnostic looked and found nothing against it. That is the best news available here, and it is not proof.',
  },
  weakened: {
    word: 'Weakened',
    means: 'A diagnostic found something that counts against it. Worth reading before you trust the estimate.',
  },
  untested: {
    word: 'Untested',
    means: 'Nothing has checked it yet. Run the linked diagnostic, or say in the report why you are not going to.',
  },
  not_applicable: {
    word: 'Does not apply',
    means: 'This design does not need it.',
  },
}

export function statusWord(s: string): string {
  return ASSUMPTION_STATUS[s]?.word ?? s
}

export function statusMeaning(s: string): string | undefined {
  return ASSUMPTION_STATUS[s]?.means
}

export function explainKeyFor(a: { id: string; explain_key?: string | null }): string {
  return a.explain_key ?? `assumption.${a.id}`
}

/** The status word as a pill. Clicking it opens the assumption's entry, and the
 *  tooltip says what the word means plus whatever the run noted about it. */
export function AssumptionPill({
  status, note, onClick, label,
}: { status: AssumptionStatus | string; note?: string | null; onClick?: () => void; label?: string }) {
  const meaning = statusMeaning(status)
  const title = [label ? `${label}: ${statusWord(status)}` : statusWord(status), meaning, note]
    .filter(Boolean).join('\n\n')
  if (!onClick) {
    return <span className={`ledger-status ${status}`} title={title}>{statusWord(status)}</span>
  }
  return (
    <button type="button" className={`ledger-status clickable ${status}`} title={title} onClick={onClick}>
      {statusWord(status)}
    </button>
  )
}

/** A list of assumptions with their statuses, each one openable.
 *
 * `showMeaning` adds the one-line description from the Explain catalogue under
 * each name, for the places where a reader meets the assumption for the first
 * time. The ledger, which already has a legend, leaves it off.
 */
export function AssumptionList({
  items, showMeaning = false, compact = false,
}: { items: Assumption[]; showMeaning?: boolean; compact?: boolean }) {
  const setExplain = useStore((s) => s.setExplain)
  const explain = useStore((s) => s.explain)
  const openLedger = useStore((s) => s.openLedger)
  const diagnosticTitles = useStore((s) => s.diagnosticTitles)
  if (!items.length) return null
  return (
    <ul className={`assumption-list${compact ? ' compact' : ''}`}>
      {items.map((a) => {
        const key = explainKeyFor(a)
        const open = () => setExplain(key)
        const short = showMeaning ? explain[key]?.short : undefined
        const diags = a.diagnostic_ids ?? []
        return (
          <li key={a.id} className="assumption-item">
            <AssumptionPill status={a.status} note={a.note} onClick={open} label={a.label ?? a.id} />
            <div className="assumption-body">
              <button type="button" className="assumption-name" onClick={open}
                      title="Read what this assumption means and how it can fail">
                {a.label ?? humaniseId(a.id)}
              </button>
              {short && <p className="assumption-short">{short}</p>}
              {a.note && <p className="assumption-note">{a.note}</p>}
              {diags.length > 0 && (
                <button type="button" className="linky tiny" onClick={() => openLedger('diagnostics')}>
                  {/* The catalogue's own names, not the storage keys. This read
                      "See the diagnostics: Love, Ess" -- two column headings
                      from a database offered as the names of two plots. */}
                  See the check{diags.length > 1 ? 's' : ''}:{' '}
                  {diags.map((d) => diagnosticTitles[d] ?? humaniseId(d)).join(', ')}
                </button>
              )}
            </div>
          </li>
        )
      })}
    </ul>
  )
}

export function humaniseId(id: string): string {
  return id.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())
}
