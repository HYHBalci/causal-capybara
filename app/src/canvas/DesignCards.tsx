/** Choose a picture, not a syllabus.
 *
 * After the question strip has a treatment and an outcome, the canvas offers
 * cards with a one-sentence gloss and a tiny schematic -- not a list of
 * estimators. People understand pictures of research designs. They do not
 * understand "select the treatment variable for matchit".
 */

import { useStore } from '../store'
import type { DesignCard } from '../types'
import { Schematic } from './Schematic'
import { count } from '../format'

export function DesignCards() {
  const designs = useStore((s) => s.designs)
  const spec = useStore((s) => s.spec)
  const setDesign = useStore((s) => s.setDesign)
  const setExplain = useStore((s) => s.setExplain)
  const setView = useStore((s) => s.setView)
  const health = useStore((s) => s.health)
  const columns = useStore((s) => s.columns)

  if (!spec) return null
  const cards = designs.filter((d) => d.id !== 'catalogue')
  const catalogue = designs.find((d) => d.id === 'catalogue')

  return (
    <div className="canvas-pad scroll">
      <header className="canvas-head">
        <h2>What kind of comparison is this?</h2>
        <p className="hint">
          Pick the picture that matches how treatment was decided. The board that follows is the
          identification strategy made visible — you complete it by dropping variables on it.
        </p>
      </header>

      {!columns.length && (
        <div className="banner caution" style={{ marginBottom: 12 }}>
          <span>
            No data imported yet. You can still choose a design and read what it assumes, but the boards
            and the live diagnostics need a dataset.
          </span>
        </div>
      )}

      <div className="design-grid">
        {cards.map((d) => (
          <DesignTile
            key={d.id}
            card={d}
            selected={spec.design === d.id}
            hint={fitHint(d, health, columns.length)}
            onChoose={() => setDesign(d.id)}
            onExplain={() => setExplain(d.explain_key)}
          />
        ))}
      </div>

      {/* This row used to open Recommend, which ranks methods against a chosen
          design and a filled board -- neither of which exists at this point, so
          the one advertised way into the catalogue sat on a spinner for ever.
          The catalogue proper needs nothing chosen. */}
      {catalogue && (
        <button className="catalogue-row card" onClick={() => setView('learn')}>
          <div>
            <div className="design-title">{catalogue.title}</div>
            <p className="design-sentence">{catalogue.sentence}</p>
          </div>
          <span className="btn ghost sm">Open the catalogue →</span>
        </button>
      )}
    </div>
  )
}

function DesignTile({
  card, selected, hint, onChoose, onExplain,
}: {
  card: DesignCard
  selected: boolean
  hint: string | null
  onChoose: () => void
  onExplain: () => void
}) {
  return (
    <div className={`design-card card${selected ? ' selected' : ''}`}>
      <button className="design-card-main" onClick={onChoose}>
        <div className="design-schematic"><Schematic kind={card.schematic} /></div>
        <div className="design-title">{card.title}</div>
        <p className="design-sentence">{card.sentence}</p>
        {hint && <p className="design-hint">{hint}</p>}
      </button>
      <div className="design-card-foot">
        <button className="btn ghost sm" onClick={onExplain}>What this assumes</button>
        {selected && <span className="chip teal">selected</span>}
      </div>
    </div>
  )
}

/** A quiet nudge from the data, never an automatic choice. */
function fitHint(card: DesignCard, health: ReturnType<typeof useStore.getState>['health'], nCols: number): string | null {
  if (!health) return null
  if (card.id === 'did') {
    if (health.staggered) return 'Your data has units adopting at different times — this design fits, and it will steer you away from two-way fixed effects.'
    if (health.n_units && health.n_periods && health.n_periods >= 2) return `${count(health.n_units)} units over ${count(health.n_periods)} periods.`
  }
  if (card.id === 'synth' && health.n_units && health.n_units < 60 && health.n_periods && health.n_periods >= 8) {
    return 'Few units, many periods — the shape synthetic control was built for.'
  }
  if (card.id === 'its' && health.n_units === 1) {
    return 'A single series: this is the interrupted-time-series shape.'
  }
  if (card.id === 'observational' && nCols > 4 && !health.n_units) {
    return 'A cross-section with covariates.'
  }
  return null
}
