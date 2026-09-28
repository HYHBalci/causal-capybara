/** Design pictures you complete.
 *
 * Each design has a board, and users complete it by dropping variables. The
 * board IS the identification strategy made visible. As soon as the minimum
 * roles exist the canvas splits: schematic on the left, live diagnostic sketches
 * on the right -- cheap, approximate, always on, and the reason a user notices
 * disaster early.
 *
 * Every drop-zone has a keyboard and list-box equivalent (see the inspector),
 * and a screen-reader label. The picture is the beginner path, not the only one.
 */

import { useState } from 'react'
import { useStore } from '../store'
import type { ColumnMeta, DesignZone } from '../types'
import { ArtifactView } from '../components/VegaChart'
import { MarkReviewedButton } from '../components/Provisional'
import { useSplitPane } from '../components/SplitPane'
import { Schematic } from './Schematic'
import { count } from '../format'

export function DesignBoard() {
  const spec = useStore((s) => s.spec)
  const designs = useStore((s) => s.designs)
  const setView = useStore((s) => s.setView)
  const profile = useStore((s) => s.profile)
  const design = designs.find((d) => d.id === spec?.design)
  const split = useSplitPane('split.board', 720, 320)

  if (!spec) return null
  if (!design || spec.design === 'undecided') {
    return (
      <div className="empty-state">
        <h2>Pick a design first</h2>
        <p className="hint">
          The board is a picture of how people came to be treated — who got the programme, and what
          decided it. It only takes shape once you have said which of those stories your data tells,
          so choose a design and the board draws itself.
        </p>
        <button className="btn primary" onClick={() => setView('design')}>Choose a design</button>
      </div>
    )
  }

  const zones = design.zones.filter((z) => profile === 'advanced' || z.profile !== 'advanced')
  const required = zones.filter((z) => z.required)
  const optional = zones.filter((z) => !z.required)

  return (
    <div className="board-split" ref={split.ref} style={split.style}>
      {split.handle}
      <div className="board-left scroll">
        <header className="canvas-head">
          <div className="row" style={{ gap: 8 }}>
            <h2>{design.title}</h2>
            <button className="btn ghost sm" onClick={() => setView('design')}>Change design</button>
          </div>
          {design.board_caption && <p className="board-caption">{design.board_caption}</p>}
        </header>

        <StaggeredNotice />

        <div className="board-figure">
          <Schematic kind={design.schematic} large />
        </div>

        <div className="zone-grid">
          {required.map((z) => <Zone key={z.role} zone={z} />)}
        </div>
        {optional.length > 0 && (
          <>
            <div className="panel-title" style={{ margin: '14px 0 6px' }}>Also on this board</div>
            <div className="zone-grid">
              {optional.map((z) => <Zone key={z.role} zone={z} />)}
            </div>
          </>
        )}

        <EstimandPicker />
        <BoardActions />
      </div>

      <div className="board-right scroll">
        <LiveSketch />
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ zone */

function Zone({ zone }: { zone: DesignZone }) {
  const spec = useStore((s) => s.spec)!
  const columns = useStore((s) => s.columns)
  const setRole = useStore((s) => s.setRole)
  const setExplain = useStore((s) => s.setExplain)
  const guardrails = useStore((s) => s.guardrails)
  const [over, setOver] = useState(false)
  const [picking, setPicking] = useState(false)

  const raw = (spec.roles as Record<string, unknown>)[zone.role]
  const list = zone.multiple ? ((raw as string[]) ?? []) : []
  const single = !zone.multiple ? (raw as string | number | null) : null
  const filled = zone.multiple ? list.length > 0 : single !== null && single !== undefined && single !== ''
  const isValue = zone.kind === 'value' || zone.kind === 'values'
  const flagged = new Set((guardrails?.bad_controls ?? []).map((b) => b.variable))

  const accepts = (c: ColumnMeta) => !zone.accepts?.length || zone.accepts.includes(c.kind)
  const options = columns.filter(accepts)

  const add = (name: string) => {
    if (zone.multiple) setRole(zone.role, [...new Set([...list, name])], `Added ${name} to ${zone.label}`)
    else setRole(zone.role, name, `Set ${zone.label} to ${name}`)
    setPicking(false)
  }

  return (
    <div
      className={`zone dropzone${filled ? ' filled' : ''}${over ? ' over' : ''}${zone.tone === 'forbidden' ? ' forbidden' : ''}`}
      onDragOver={(e) => {
        if (isValue) return
        if (e.dataTransfer.types.includes('text/capy-variable')) { e.preventDefault(); setOver(true) }
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const name = e.dataTransfer.getData('text/capy-variable')
        if (name) add(name)
      }}
      role="group"
      aria-label={`${zone.label}${zone.required ? ' (required)' : ''}`}
    >
      <div className="zone-head row">
        <span className="zone-label">{zone.label}</span>
        {zone.required && !filled && <span className="chip ochre tiny">needed</span>}
        <div className="spacer" />
        {!isValue && (
          <button className="btn ghost sm" onClick={() => setPicking((p) => !p)}
                  aria-expanded={picking} aria-label={`Choose a variable for ${zone.label}`}>
            {filled && !zone.multiple ? 'change' : 'add'}
          </button>
        )}
      </div>

      {isValue ? (
        <input
          className="zone-value"
          type="text"
          value={single === null || single === undefined ? '' : String(single)}
          placeholder={zone.role === 'cutoff' ? 'e.g. 0' : 'value'}
          aria-label={zone.label}
          onChange={(e) => {
            const t = e.target.value
            const n = Number(t)
            setRole(
              zone.role,
              t === '' ? null : Number.isFinite(n) && t.trim() !== '' ? n : t,
              t === '' ? `Cleared ${zone.label}` : `Set ${zone.label} to ${t}`,
            )
          }}
        />
      ) : zone.multiple ? (
        <div className="row wrap zone-chips">
          {list.map((v) => (
            <button key={v} className={`chip role${flagged.has(v) ? ' crimson' : ''}`}
                    title={flagged.has(v)
                      ? (guardrails?.bad_controls ?? []).find((b) => b.variable === v)?.reason
                      : 'Remove'}
                    onClick={() => setRole(zone.role, list.filter((x) => x !== v), `Removed ${v} from ${zone.label}`)}>
              {v} ✕
            </button>
          ))}
          {/* The only place a variable can be picked up and dragged is the list
              on the Data screen, which is not on this one, so the empty zone
              points at the button that is actually here. */}
          {!list.length && <span className="hint tiny">Empty — use “add” above to choose columns</span>}
        </div>
      ) : (
        <div className="zone-single">
          {filled ? (
            <button className="chip role" onClick={() => setRole(zone.role, null, `Cleared ${zone.label}`)}>
              {String(single)} ✕
            </button>
          ) : (
            <span className="hint tiny">Empty — use “add” above to choose a column</span>
          )}
        </div>
      )}

      {picking && !isValue && (
        <div className="zone-picker card">
          <div className="scroll" style={{ maxHeight: 220 }}>
            {options.map((c) => (
              <button key={c.name} className="qslot-option" onClick={() => add(c.name)}>
                <span className={`kind-icon ${c.kind}`} aria-hidden />
                <span className="grow">{c.name}</span>
                <span className="tiny hint">{c.kind}</span>
              </button>
            ))}
            {!options.length && (
              <div className="pad hint tiny">
                Nothing in this dataset fits here. This box takes {acceptedKinds(zone)}, and no column
                in the data was read that way. The Data screen lets you change how a column was read.
              </div>
            )}
          </div>
        </div>
      )}

      {zone.help && <p className="zone-help tiny hint">{zone.help}</p>}
      {zone.role === 'confounders' && (
        <button className="linky tiny" onClick={() => setExplain('guardrail.bad_control')}>
          What must not go in here
        </button>
      )}
    </div>
  )
}

/** The kinds a zone accepts, in words rather than in the storage vocabulary.
 *  "binary" and "continuous" are names for how a column was read, and a reader
 *  who has never opened a statistics package has no way to guess them. */
const KIND_WORDS: Record<string, string> = {
  binary: 'yes/no columns',
  continuous: 'number columns',
  categorical: 'columns with a handful of named groups',
  datetime: 'date columns',
  id: 'identifier columns',
  text: 'free-text columns',
}

function acceptedKinds(zone: DesignZone): string {
  const words = (zone.accepts ?? []).map((k) => KIND_WORDS[k] ?? k)
  if (!words.length) return 'any column'
  if (words.length === 1) return words[0]
  return `${words.slice(0, -1).join(', ')} or ${words[words.length - 1]}`
}

/* ------------------------------------------------------- staggered notice */

function StaggeredNotice() {
  const spec = useStore((s) => s.spec)!
  const designs = useStore((s) => s.designs)
  const sketch = useStore((s) => s.sketch)
  const setExplain = useStore((s) => s.setExplain)
  const design = designs.find((d) => d.id === spec.design)
  const staggered = sketch?.values?.staggered === true
  if (!design?.staggered_copy || !staggered) return null
  return (
    <div className="banner warning board-notice">
      <strong>{design.staggered_copy}</strong>
      <span>
        {' '}Your units did not all start at the same time, and that quietly changes the arithmetic.
        The ordinary before-and-after regression ends up comparing units that were treated earlier
        against units that were already treated — a comparison that answers no question you asked, and
        that can pull the average in the wrong direction. Two of the recommended methods are built to
        avoid it, and the app will put them at the top of the list.
      </span>
      <button className="linky" onClick={() => setExplain('concept.staggered_adoption')}>
        Why different start dates matter
      </button>
    </div>
  )
}

/* ------------------------------------------------------- estimand picker */

function EstimandPicker() {
  const spec = useStore((s) => s.spec)!
  const catalog = useStore((s) => s.estimandCatalog)
  const designs = useStore((s) => s.designs)
  const profile = useStore((s) => s.profile)
  const setEstimand = useStore((s) => s.setEstimand)
  const setExplain = useStore((s) => s.setExplain)
  const design = designs.find((d) => d.id === spec.design)
  if (!design) return null

  const allowed = new Set(design.estimands ?? [])
  const shown = catalog.filter((e) =>
    profile === 'advanced' ? true : profile === 'standard' ? e.profile !== 'advanced' : e.profile === 'beginner',
  )

  return (
    <section className="estimand-block">
      <div className="panel-title">What exactly do you want a number for?</div>
      <p className="tiny hint" style={{ margin: '2px 0 8px' }}>
        Each line is a different question, and they can have different answers on the same data — the
        effect on everybody is not the effect on the people who actually took the programme. Choosing
        one before you choose a method is the whole trick; the technical name for the number you pick
        is the <em>estimand</em>.{' '}
        <button className="linky" onClick={() => setExplain('concept.estimand_first')}>
          Why this comes first
        </button>
      </p>
      <div className="estimand-list" role="radiogroup" aria-label="What you want a number for">
        {shown.map((e) => {
          const available = allowed.has(e.id)
          const checked = spec.estimand === e.id
          return (
            <label key={e.id} className={`estimand-row${checked ? ' on' : ''}${available ? '' : ' off'}`}>
              <input
                type="radio"
                name="estimand"
                checked={checked}
                disabled={!available}
                onChange={() => setEstimand(e.id)}
              />
              <span className="grow">
                <span className="estimand-sentence">{e.sentence}</span>
                {!available && (
                  <span className="tiny hint">
                    {' '}The “{design.title}” design cannot answer this one — nothing in it pins this
                    number down, whatever method you run.
                  </span>
                )}
              </span>
              <button
                className="chip"
                title={`Read what “${e.acronym}” means`}
                onClick={(ev) => { ev.preventDefault(); setExplain(e.explain_key) }}
              >
                {e.acronym}
              </button>
            </label>
          )
        })}
      </div>
    </section>
  )
}

/* -------------------------------------------------------- board actions */

function BoardActions() {
  const guardrails = useStore((s) => s.guardrails)
  const setView = useStore((s) => s.setView)
  const setExplain = useStore((s) => s.setExplain)
  const spec = useStore((s) => s.spec)!
  const missing = guardrails?.missing_roles ?? []

  return (
    <div className="board-actions">
      {missing.length > 0 ? (
        <div className="banner info">
          <span>Still needed: {missing.map((m) => m.label).join(', ')}.</span>
        </div>
      ) : (
        <div className="banner info">
          <span>
            The board is complete. Have a look at the check on the next screen before you estimate. You
            can skip it, but then every result is stamped not-yet-checked — “provisional” — in the
            results, in the project list and in the report, until you come back and look.{' '}
            <button className="linky" onClick={() => setExplain('concept.provisional')}>
              What provisional means
            </button>
          </span>
        </div>
      )}
      <div className="row" style={{ gap: 8, marginTop: 8 }}>
        <button className="btn" onClick={() => setView('diagnose')} disabled={missing.length > 0}>
          Diagnose
        </button>
        <button className="btn primary" onClick={() => setView('recommend')}
                disabled={missing.length > 0 || !spec.estimand}>
          Choose methods →
        </button>
      </div>
    </div>
  )
}

/* ----------------------------------------------------------- live sketch */

export function LiveSketch({ full = false, showReview = true }: { full?: boolean; showReview?: boolean }) {
  const sketch = useStore((s) => s.sketch)
  const loading = useStore((s) => s.sketchLoading)
  const setExplain = useStore((s) => s.setExplain)
  const guardrails = useStore((s) => s.guardrails)

  if (loading && !sketch) {
    return <div className="sketch-empty row"><span className="spinner" /> Drawing the first check…</div>
  }
  if (!sketch) {
    return (
      <div className="sketch-empty hint">
        A picture of your data appears here — the one check worth doing before any estimate — as soon as
        the board has the boxes marked “needed”. Fill those in on the left and it draws itself.
      </div>
    )
  }

  // The sketch names the boxes it is still waiting for by their storage id
  // ("treated_unit"). The guardrails carry the same list with the labels the
  // board prints, so the two panels say the same words.
  const roleLabel = (role: string) =>
    guardrails?.missing_roles?.find((m) => m.role === role)?.label ?? role.replace(/_/g, ' ')

  return (
    <section className="sketch">
      <div className="row sketch-head">
        <span className="panel-title">{sketch.title ?? 'A first look at your data'}</span>
        {loading && <span className="spinner" />}
        <div className="spacer" />
        {sketch.elapsed_ms !== undefined && (
          <span className="tiny hint"
                title={sketch.subsampled
                  ? 'To keep this instant it was drawn from a random sample of your rows, not all of '
                    + 'them. The full diagnostics that come with an estimate use every row.'
                  : undefined}>
            {Math.round(sketch.elapsed_ms)} ms{sketch.subsampled ? ' · drawn from a sample of the rows' : ''}
          </span>
        )}
      </div>

      {sketch.summary && <p className="sketch-summary">{sketch.summary}</p>}

      {sketch.missing_roles?.length > 0 && (
        <div className="banner info">
          <span>Still needed: {sketch.missing_roles.map(roleLabel).join(', ')}.</span>
        </div>
      )}

      {sketch.warnings?.map((w, i) => (
        <div key={i} className={`banner ${w.level}`} style={{ marginBottom: 8 }}>
          <span>{w.message}</span>
        </div>
      ))}

      {sketch.artifacts?.map((a) => (
        <ArtifactView key={a.id} artifact={a} onExplain={setExplain} height={full ? 260 : 190} />
      ))}

      {sketch.ready && sketch.id && showReview && (
        <div style={{ marginTop: 10 }}>
          <MarkReviewedButton sketchId={sketch.id} compact={!full} />
        </div>
      )}

      {Object.keys(sketch.values ?? {}).length > 0 && (
        <details className="sketch-values">
          <summary className="tiny hint">The numbers behind this picture</summary>
          <dl className="kv tiny">
            {Object.entries(sketch.values).map(([k, v]) => (
              <div key={k} style={{ display: 'contents' }}>
                <dt>{valueLabel(k)}</dt>
                <dd className="num">{formatValue(v)}</dd>
              </div>
            ))}
          </dl>
        </details>
      )}
      <p className="tiny hint" style={{ marginTop: 8 }}>
        A quick look, not the full check. The complete set of checks comes with each estimate.
      </p>
    </section>
  )
}

/** The keys the sketches emit are storage names. Swapping the underscores for
 *  spaces turned "pct_outside_common_support" into "pct outside common support",
 *  which is not English, so the ones that actually appear are named here and the
 *  rest fall back to a readable expansion of the two common prefixes. */
const SKETCH_VALUE_LABELS: Record<string, string> = {
  n: 'Rows',
  n_treated: 'Treated',
  n_control: 'Untreated',
  n_units: 'Units',
  n_periods: 'Time periods',
  n_pre: 'Periods before',
  n_post: 'Periods after',
  n_donors: 'Possible comparison units',
  n_never_treated: 'Never treated',
  n_cohorts: 'Different start dates',
  n_mediators: 'Steps in between',
  n_left: 'Below the cutoff',
  n_right: 'At or above the cutoff',
  n_mass_points: 'Distinct scores',
  min_share_treated: 'Smallest share treated',
  max_share_treated: 'Largest share treated',
  pct_near_cutoff: 'Share close to the cutoff',
  pct_outside_common_support: 'Share with nobody comparable',
  person_time: 'Person-periods',
  first_stage_t: 'Strength of the instrument',
  staggered: 'Started at different times',
  overlap: 'Overlap between the groups',
}

function valueLabel(key: string): string {
  const known = SKETCH_VALUE_LABELS[key]
  if (known) return known
  const words = key.replace(/^n_/, 'number of ').replace(/^pct_/, 'share of ').replace(/_/g, ' ')
  return words.replace(/^./, (c) => c.toUpperCase())
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (Array.isArray(v)) return v.slice(0, 8).map((x) => formatValue(x)).join(', ') + (v.length > 8 ? ' …' : '')
  if (typeof v === 'number') return Number.isInteger(v) ? count(v) : v.toPrecision(4)
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (typeof v === 'object') return JSON.stringify(v).slice(0, 80)
  return String(v)
}
