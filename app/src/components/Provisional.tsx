/** What "provisional" means, and what to do about it.
 *
 * The flag was reported and never explained. The results screen said
 * "Provisional. The overlap diagnostic for this design had not been viewed when
 * the run started" and stopped there -- no name you could click, no route to
 * the diagnostic, and no hint that reviewing it afterwards leaves the finished
 * runs flagged until they are run again. People read it as a defect in the app.
 *
 * Two different things also arrived in one paragraph. "You never looked at the
 * overlap plot" is a procedural flag you can clear. "The weights are
 * degenerate" is a finding about the data, and no amount of reviewing will
 * answer it. They are separated here.
 */

import { useStore } from '../store'
import type { RunResult } from '../types'

/** The banner on the results screen, with the action that clears the flag. */
export function ProvisionalNotice({ results }: { results: RunResult[] }) {
  const guardrails = useStore((s) => s.guardrails)
  const setView = useStore((s) => s.setView)
  const runMethods = useStore((s) => s.runMethods)
  const setSelectedRuns = useStore((s) => s.setSelectedRuns)
  const spec = useStore((s) => s.spec)
  const reviewed = useReviewed(guardrails?.core_diagnostic)

  const flagged = results.filter((r) => r.provisional)
  const clean = results.filter((r) => !r.provisional)
  if (!flagged.length) return null

  // A flagged run is superseded once the same method has been estimated again
  // without the flag. Re-running a second time would only add another pair.
  const superseded = flagged.filter((f) => clean.some((c) => c.method === f.method))
  const allSuperseded = superseded.length === flagged.length

  // The sentence the review would have removed, told apart from the rest.
  const reviewReasons = new Set(
    flagged.map((r) => r.review_reason).filter((x): x is string => !!x),
  )
  const needsReview = flagged.filter((r) => r.review_diagnostic)
  const otherReasons = [...new Set(
    flagged.flatMap((r) => r.provisional_reasons ?? []).filter((x) => !reviewReasons.has(x)),
  )]

  const label = guardrails?.core_diagnostic_label ?? 'the core diagnostic'
  const reviewedSince = reviewed
  const methodIds = [...new Set(flagged.map((r) => r.method))]

  return (
    <div className="banner caution provisional-notice">
      <div className="col grow" style={{ gap: 6 }}>
        <div>
          <strong>
            {flagged.length === results.length
              ? flagged.length === 1 ? 'This run is provisional.' : 'These runs are provisional.'
              : `${flagged.length} of ${results.length} runs are provisional.`}
          </strong>{' '}
          <span>
            Review the reasons below before interpreting these estimates. These cautions are
            saved with the runs and included in the report.
          </span>
        </div>

        {needsReview.length > 0 && (
          allSuperseded ? (
            <div className="row provisional-action">
              <span className="grow">
                You have already estimated {methodIds.length === 1 ? 'this method' : 'these methods'} again
                since reviewing <strong>{label}</strong>, and the newer{' '}
                {clean.length === 1 ? 'run is' : 'runs are'} not flagged. The earlier{' '}
                {flagged.length === 1 ? 'one is' : 'ones are'} still here, and still flagged, because
                that is what happened.
              </span>
              <button
                className="btn sm"
                onClick={() => setSelectedRuns(clean.map((r) => r.run_id))}
              >
                Hide the earlier {flagged.length === 1 ? 'run' : 'runs'}
              </button>
            </div>
          ) : reviewedSince ? (
            <div className="row provisional-action">
              <span className="grow">
                You have since reviewed <strong>{label}</strong>. These runs still carry the flag
                because they were estimated before that. Estimating again removes the missing-review
                flag; any warnings about the data or method still apply.
              </span>
              <button
                className="btn sm primary"
                disabled={!spec || !methodIds.length}
                onClick={() => void runMethods(methodIds)}
              >
                Estimate {methodIds.length === 1 ? 'it' : 'them'} again
              </button>
            </div>
          ) : (
            <div className="row provisional-action">
              <span className="grow">
                They were estimated before <strong>{label}</strong> had been looked at. Open it, and
                mark it reviewed there; later runs are then not flagged for this reason.
              </span>
              <button className="btn sm primary" onClick={() => setView('diagnose')}>
                Review it now →
              </button>
            </div>
          )
        )}

        {otherReasons.length > 0 && (
          <div>
            <span className="provisional-also">
              {needsReview.length > 0 ? 'Also flagged, and not something reviewing can settle:' : 'Flagged because:'}
            </span>{' '}
            <span>{otherReasons.join(' ')}</span>
          </div>
        )}
      </div>
    </div>
  )
}

/** The button that records the review, shown wherever the diagnostic is.
 *
 * The only copy of this used to sit at the foot of the sketch panel, under a
 * chart, where a reader who had scrolled to the plot never saw it.
 */
/** Whether this diagnostic has been reviewed.
 *
 * Read from the spec in hand rather than from the guardrails, which are
 * recomputed by the engine from the saved file. Waiting for that round trip
 * meant the button did not change when it was pressed.
 */
function useReviewed(sketchId: string | null | undefined): boolean {
  const viewed = useStore((s) => s.spec?.diagnostics_viewed)
  const fromEngine = useStore((s) => s.guardrails?.core_diagnostic_viewed)
  if (!sketchId) return !!fromEngine
  return (viewed ?? []).includes(sketchId) || !!fromEngine
}

export function MarkReviewedButton({ sketchId, compact }: { sketchId: string; compact?: boolean }) {
  const guardrails = useStore((s) => s.guardrails)
  const markViewed = useStore((s) => s.markDiagnosticViewed)
  const label = guardrails?.core_diagnostic_label ?? 'this diagnostic'
  const reviewed = useReviewed(sketchId)

  if (reviewed) {
    return (
      <div className="row provisional-reviewed">
        <span className="chip moss">Reviewed</span>
        {!compact && (
          <span className="tiny hint">
            Recorded with the project. Runs from here on are not flagged for this reason.
          </span>
        )}
      </div>
    )
  }
  return (
    <div className={`provisional-review-cta${compact ? ' compact' : ''}`}>
      <button className="btn primary" onClick={() => markViewed(sketchId)}>
        I have reviewed {compact ? 'this' : `“${label}”`}
      </button>
      {!compact && (
        <p className="tiny hint" style={{ margin: '5px 0 0' }}>
          Records that you looked before estimating. Without it, every run from this question is
          marked provisional — including ones you have already made.
        </p>
      )}
    </div>
  )
}
