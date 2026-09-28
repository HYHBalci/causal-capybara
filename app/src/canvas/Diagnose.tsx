/** Diagnose before you estimate.
 *
 * Estimate stays enabled -- we never trap a user -- but it is not enabled
 * *quietly*: the core diagnostic for the design has to have been viewed, and
 * "viewed" is provenance, not a checkbox. Proceeding anyway marks the run
 * provisional, and that flag travels into the printout and the report.
 */

import { useEffect, useMemo } from 'react'
import { useStore } from '../store'
import type { Assumption } from '../types'
import { AssumptionList } from '../components/Assumptions'
import { MarkReviewedButton } from '../components/Provisional'
import { useSplitPane } from '../components/SplitPane'
import { LiveSketch } from './DesignBoard'

export function Diagnose() {
  const spec = useStore((s) => s.spec)
  const designs = useStore((s) => s.designs)
  const guardrails = useStore((s) => s.guardrails)
  const refreshSketch = useStore((s) => s.refreshSketch)
  const setView = useStore((s) => s.setView)
  const setExplain = useStore((s) => s.setExplain)
  const health = useStore((s) => s.health)
  const design = designs.find((d) => d.id === spec?.design)
  const split = useSplitPane('split.diagnose', 700, 300)

  useEffect(() => { void refreshSketch() }, [spec?.id, spec?.design])

  if (!spec || !design) {
    return (
      <div className="empty-state">
        <h2>Choose a design first</h2>
        <p className="hint">
          This screen runs the one check that is worth doing before you spend any time estimating — the
          check that tends to catch the trouble. Which check that is depends on how people came to be
          treated, so it needs a design before it has anything to show.
        </p>
        <button className="btn primary" onClick={() => setView('design')}>Choose a design</button>
      </div>
    )
  }

  const missing = guardrails?.missing_roles ?? []

  return (
    <div className="canvas-pad scroll">
      <header className="canvas-head">
        <h2>Before you estimate</h2>
        <p className="hint">
          The one check worth doing first for the “{design.title}” design. It is quick and rough — the
          full set of checks comes with each estimate — and its job is to show you trouble before you
          spend time on numbers that trouble would have spoiled.
        </p>
      </header>

      {missing.length > 0 && (
        <div className="banner caution">
          <span>
            The board still has empty boxes ({missing.map((m) => m.label).join(', ')}), so this check
            can only look at part of the picture.{' '}
            <button className="linky" onClick={() => setView('board')}>Finish the board</button>
          </span>
        </div>
      )}

      <div className="diagnose-split" ref={split.ref} style={split.style}>
        {split.handle}
        <div className="card pad">
          {/* The call to action lives in the Next card, which stays in view; a
              second copy under a tall chart is just noise. */}
          <LiveSketch full showReview={false} />
        </div>

        <aside className="col" style={{ gap: 10 }}>
          <section className="card pad">
            <div className="panel-title">What this design assumes</div>
            <p className="tiny hint" style={{ margin: '4px 0 8px' }}>
              Choosing a design does not make any of these true. Each one is something you have to be
              able to argue for about your own data — no amount of arithmetic can settle it. Press a
              name to read what it means and what would break it.
            </p>
            <AssumptionPreview designId={design.id} />
          </section>

          {health?.warnings?.length ? (
            <section className="card pad">
              <div className="panel-title">From the data itself</div>
              {health.warnings.map((w, i) => (
                <div key={i} className={`banner ${w.level}`} style={{ marginBottom: 6 }}>
                  <span>{w.message}</span>
                </div>
              ))}
            </section>
          ) : null}

          <section className="card pad">
            <div className="panel-title">Next</div>
            {/* The review button also lives at the foot of the sketch, which on
                a tall chart is below the fold. This copy is always in view. */}
            {guardrails?.core_diagnostic ? (
              <>
                {guardrails.core_diagnostic_viewed ? (
                  <p className="tiny hint" style={{ margin: '4px 0 8px' }}>
                    You have looked at this one. Estimates from here on carry no mark for it. Anything
                    you ran earlier keeps its mark until you run it again.
                  </p>
                ) : (
                  <p className="tiny hint" style={{ margin: '4px 0 8px' }}>
                    Look at the picture on the left first. You are allowed to estimate without it, but
                    every result you get is then stamped not-yet-checked — “provisional” — wherever it
                    appears.{' '}
                    <button className="linky" onClick={() => setExplain('concept.provisional')}>
                      What provisional means
                    </button>
                  </p>
                )}
                <MarkReviewedButton sketchId={guardrails.core_diagnostic} />
                <div className="row" style={{ marginTop: 10 }}>
                  <button className="btn" onClick={() => setView('recommend')}>Choose methods →</button>
                </div>
              </>
            ) : (
              <>
                {/* No single check stands out for this design, so nothing here
                    can be marked reviewed and nothing will be flagged for it. */}
                <p className="tiny hint" style={{ margin: '4px 0 8px' }}>
                  This design has no single check that has to come first, so there is nothing to sign
                  off here. Read the picture on the left, then go and pick the methods.
                </p>
                <button className="btn primary" onClick={() => setView('recommend')}>Choose methods →</button>
              </>
            )}
          </section>
        </aside>
      </div>
    </div>
  )
}

function AssumptionPreview({ designId }: { designId: string }) {
  const catalog = useStore((s) => s.assumptions)
  const designs = useStore((s) => s.designs)
  const design = designs.find((d) => d.id === designId)
  // Catalogue labels and default statuses, in the order the design lists them.
  // The list used to print the raw ids ("sutva", "no_manipulation").
  const items: Assumption[] = useMemo(() => {
    const ids = design?.assumptions ?? []
    return ids.map((id) => {
      const entry = catalog.find((a) => a.id === id)
      return entry
        ? { id, label: entry.label, status: entry.default_status, explain_key: entry.explain_key }
        : { id, label: id.replace(/_/g, ' '), status: 'untested' as const }
    })
  }, [catalog, design])
  if (!design) return null
  // AssumptionList draws nothing for an empty list, which left the panel a
  // heading over blank space; a design with no listed assumptions is a fact
  // worth stating rather than a gap.
  if (!items.length) {
    return (
      <p className="tiny hint">
        The catalogue lists no assumptions for this design yet. That is a gap in our notes, not a
        licence: read what the methods on the next screen assume before you trust any of them.
      </p>
    )
  }
  return <AssumptionList items={items} showMeaning />
}
