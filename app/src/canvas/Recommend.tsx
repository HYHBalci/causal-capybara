/** Recommend, don't dump a toolbox.
 *
 * After the estimand and the roles exist, 1-3 recommended methods and a longer
 * "also reasonable" list. Greyed cards for inappropriate methods say why, and
 * can still be included on purpose -- it is often useful to show the wrong
 * number next to the right one. Including a disrecommended method auto-adds a
 * warning to the report.
 *
 * The default button estimates the recommended SET, not the first card.
 */

import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { References } from '../components/References'
import type { MethodOption, RecoCard } from '../types'

/** Which program does the arithmetic, said in words. "python" and "r" are the
 *  names of the engines, and the history list is read by people who chose
 *  "Both, and compare them" from a menu, not by people who typed an engine id. */
const ENGINE_WORDS: Record<string, string> = {
  python: 'the built-in calculator',
  r: 'R',
  both: 'both programs, to compare them',
}

export function Recommend() {
  const spec = useStore((s) => s.spec)
  const reco = useStore((s) => s.recommendation)
  const error = useStore((s) => s.recommendationError ?? s.saveError)
  const loading = useStore((s) => s.recommendationLoading)
  const refresh = useStore((s) => s.refreshRecommendation)
  const runMethods = useStore((s) => s.runMethods)
  const updateSpec = useStore((s) => s.updateSpec)
  const setView = useStore((s) => s.setView)
  const guardrails = useStore((s) => s.guardrails)
  const [search, setSearch] = useState('')

  useEffect(() => { if (spec && !reco) void refresh() }, [spec?.id])

  const included = useMemo(
    () => new Set((spec?.methods ?? []).filter((m) => m.included !== false).map((m) => m.method_id)),
    [spec?.methods],
  )

  if (!spec || spec.design === 'undecided') return <div className="empty-state">
    <h2>Choose a research design first</h2>
    <p className="hint">Recommendations need a question and a design. You can also read about any method in the catalogue.</p>
    <button className="btn primary" onClick={() => setView(spec ? 'design' : 'data')}>{spec ? 'Choose a design' : 'Open data'}</button>
    <button className="btn ghost" onClick={() => setView('learn')}>Browse the method catalogue</button>
  </div>
  if (!reco) {
    return <div className="empty-state">
      {error ? <>
        <h2>Methods could not be loaded</h2>
        <p className="hint" role="alert">{error}</p>
        <button className="btn primary" disabled={loading} onClick={() => void refresh()}>Try again</button>
        <button className="btn ghost" onClick={() => setView('board')}>Back to the board</button>
      </> : loading ? <p className="row" role="status"><span className="spinner" /> Working out what fits this question…</p>
        : <><h2>Check which methods fit this question</h2><button className="btn primary" onClick={() => void refresh()}>Load methods</button></>}
    </div>
  }
  // Nothing recommended and nothing reasonable means the board is not far enough
  // along for the engine to judge, and a page of three empty headings says so
  // much less clearly than one sentence and the way back.
  if (!reco.recommended.length && !reco.reasonable.length && !reco.unsuitable.length) {
    return (
      <div className="empty-state">
        <h2>Not enough to go on yet</h2>
        <p className="hint">
          Which methods fit depends on the design, on the exact number you asked for, and on what the
          data turns out to look like. One of those is still missing, so there is nothing worth
          recommending yet. Go back to the board and fill in the boxes marked “needed”.
        </p>
        <button className="btn primary" onClick={() => setView('board')}>Back to the board</button>
      </div>
    )
  }

  const toggle = (card: RecoCard) => {
    const already = included.has(card.id)
    updateSpec(
      (s) => {
        const methods = [...(s.methods ?? [])]
        const i = methods.findIndex((m) => m.method_id === card.id)
        if (already) {
          if (i >= 0) methods.splice(i, 1)
        } else if (i >= 0) {
          methods[i] = { ...methods[i], included: true }
        } else {
          methods.push({
            method_id: card.id,
            engine: useStore.getState().methodHealth[card.id]?.python?.available === false
              ? 'r' : 'python',
            options: {},
            included: true,
            acknowledged_disrecommended: card.status === 'disrecommended',
          })
        }
        return { ...s, methods }
      },
      already ? `Removed ${card.title} from the comparison` : `Included ${card.title} in the comparison`,
      { method_id: card.id, status: card.status },
    )
  }

  // The history list is prose the analyst reads back later, so it names the
  // method and the setting the way the card does rather than by their ids.
  const setEngine = (card: RecoCard, engine: 'python' | 'r' | 'both') => {
    updateSpec((s) => {
      const methods = [...(s.methods ?? [])]
      const i = methods.findIndex((m) => m.method_id === card.id)
      if (i < 0) return s
      methods[i] = { ...methods[i], engine }
      return { ...s, methods }
    }, `Chose to run ${card.title} on ${ENGINE_WORDS[engine]}`, { method_id: card.id, engine })
  }

  const setOption = (card: RecoCard, option: MethodOption, value: unknown) => {
    updateSpec((s) => {
      const methods = [...(s.methods ?? [])]
      const i = methods.findIndex((m) => m.method_id === card.id)
      if (i < 0) return s
      methods[i] = { ...methods[i], options: { ...(methods[i].options ?? {}), [option.name]: value } }
      return { ...s, methods }
    }, `Set “${option.label}” on ${card.title} to ${value}`, { method_id: card.id, [option.name]: value })
  }

  const defaultSet = reco.default_set ?? []
  const chosen = [...included]
  const runSet = chosen.length ? chosen : defaultSet
  const filter = (c: RecoCard) =>
    !search || c.title.toLowerCase().includes(search.toLowerCase()) ||
    (c.one_liner ?? '').toLowerCase().includes(search.toLowerCase())
  const noMatches = !!search.trim() &&
    ![...reco.recommended, ...reco.reasonable, ...reco.unsuitable].some(filter)

  return (
    <div className="canvas-pad scroll">
      <header className="canvas-head row" style={{ alignItems: 'flex-start' }}>
        <div className="grow">
          <h2>Which methods should answer this?</h2>
          <p className="hint">
            These come from three things: how people came to be treated, the exact number you asked for
            on the board, and what your data actually looks like. The normal thing to do is run the
            whole recommended set and see whether they agree — not to pick one and hope.
          </p>
        </div>
        <input placeholder="Search the catalogue" value={search}
               onChange={(e) => setSearch(e.target.value)} aria-label="Search methods" />
      </header>

      {guardrails?.would_be_provisional && (
        <div className="banner caution">
          <span className="grow">
            You have not looked at{' '}
            <strong>{guardrails.core_diagnostic_label ?? 'the check that comes first for this design'}</strong>{' '}
            yet. You may estimate now, but every result will be stamped not-yet-checked —
            “provisional” — in the results, in the project list and in the report, until you go and look
            and then run them again.
          </span>
          <button className="btn sm" onClick={() => setView('diagnose')}>Look at it first →</button>
        </div>
      )}

      <div className="run-bar card">
        <div className="grow">
          <strong>
            {runSet.length
              ? `${runSet.length} method${runSet.length > 1 ? 's' : ''} selected`
              : 'Nothing selected yet'}
          </strong>
          <span className="hint">
            {' — '}
            {runSet.length
              ? (chosen.length ? 'your choice' : 'the set we recommend')
              : 'press “Include in comparison” on a card below'}
          </span>
        </div>
        <button className="btn ghost" onClick={() => setView('diagnose')}>Run the first check instead</button>
        <button
          className="btn primary"
          disabled={!runSet.length}
          onClick={() => void runMethods(runSet, Object.fromEntries(
            (spec.methods ?? []).map((m) => [m.method_id, m.options ?? {}]),
          ))}
        >
          Estimate the {chosen.length ? 'selected' : 'recommended'} set
        </button>
      </div>

      <Section title="Recommended here" cards={reco.recommended.filter(filter)} tone="recommended"
               included={included} onToggle={toggle} onOption={setOption}
               onEngine={setEngine} spec={spec} />
      <Section title="Also reasonable" cards={reco.reasonable.filter(filter)} tone="reasonable"
               included={included} onToggle={toggle} onOption={setOption}
               onEngine={setEngine} spec={spec} />
      <Section title="Not appropriate here" cards={reco.unsuitable.filter(filter)} tone="unsuitable"
               included={included} onToggle={toggle} onOption={setOption}
               onEngine={setEngine} spec={spec} />

      {/* Searching for something the catalogue does not have hid all three
          sections at once and left the page apparently broken. */}
      {noMatches && (
        <div className="empty-state">
          <h2>No method here is called “{search}”</h2>
          <p className="hint">
            The search looks at method names and their one-line descriptions. Clear it to see everything
            that fits this question, or look the term up in Learn, which covers far more than the
            methods this app can run.
          </p>
          <button className="btn primary" onClick={() => setSearch('')}>Clear the search</button>
        </div>
      )}
    </div>
  )
}

function Section({
  title, cards, tone, included, onToggle, onOption, onEngine, spec,
}: {
  title: string
  cards: RecoCard[]
  tone: 'recommended' | 'reasonable' | 'unsuitable'
  included: Set<string>
  onToggle: (c: RecoCard) => void
  onOption: (card: RecoCard, option: MethodOption, value: unknown) => void
  onEngine: (card: RecoCard, engine: 'python' | 'r' | 'both') => void
  spec: NonNullable<ReturnType<typeof useStore.getState>['spec']>
}) {
  if (!cards.length) return null
  return (
    <section className="method-section">
      <div className="panel-title">{title}</div>
      {tone === 'unsuitable' && (
        <p className="tiny hint">
          You can still include one on purpose. Showing the wrong number next to the right one teaches more
          than hiding it — and the report will carry the warning.
        </p>
      )}
      <div className="method-list">
        {cards.map((c) => (
          <MethodCardView
            key={c.id}
            card={c}
            tone={tone}
            included={included.has(c.id)}
            options={(spec.methods ?? []).find((m) => m.method_id === c.id)?.options ?? {}}
            engine={(spec.methods ?? []).find((m) => m.method_id === c.id)?.engine ?? 'python'}
            onToggle={() => onToggle(c)}
            onOption={(o, v) => onOption(c, o, v)}
            onEngine={(e) => onEngine(c, e)}
          />
        ))}
      </div>
    </section>
  )
}

function MethodCardView({
  card, tone, included, options, engine, onToggle, onOption, onEngine,
}: {
  card: RecoCard
  tone: string
  included: boolean
  options: Record<string, unknown>
  engine: string
  onToggle: () => void
  onOption: (option: MethodOption, value: unknown) => void
  onEngine: (engine: 'python' | 'r' | 'both') => void
}) {
  const profile = useStore((s) => s.profile)
  const engines = useStore((s) => s.engines)
  const health = useStore((s) => s.methodHealth[card.id])
  const setExplain = useStore((s) => s.setExplain)
  const [open, setOpen] = useState(false)

  // Ask the engines what they can run rather than inferring it from the card's
  // `engines.r` field, which names the R package a researcher would otherwise
  // reach for. An adapter written in base R needs nothing installed, and a card
  // naming a package is not a claim that the engine wraps it.
  const rMissing = health?.r?.missing ?? []
  const rHealthy = !!health?.r?.available
  const pyAvailable = health?.python?.available ?? card.python_available
  const shownOptions = card.options.filter(
    (o) => profile === 'advanced' || (profile === 'standard' && o.profile === 'standard'),
  )

  return (
    <article className={`method-card card tone-${tone}${included ? ' included' : ''}`}>
      <div className="method-head row">
        <div className="grow">
          <div className="row" style={{ gap: 6 }}>
            <h3 className="method-title">{card.title}</h3>
            {card.status === 'disrecommended' && <span className="chip ochre">not recommended here</span>}
            {card.blocked && <span className="chip">needs more of the board filled in</span>}
          </div>
          <p className="method-one-liner">{card.one_liner}</p>
        </div>
        <div className="method-engines">
          <span className={`engine-pill ${pyAvailable ? 'on' : 'off'}`}
                title={health?.python?.detail ?? (pyAvailable
                  ? 'The built-in calculator can run this method. Nothing to install.'
                  : 'The built-in calculator cannot run this method yet.')}>
            <span className={`dot ${pyAvailable ? 'healthy' : 'unavailable'}`} />Py
          </span>
          <span className={`engine-pill ${rHealthy ? 'on' : 'off'}`}
                title={health?.r?.detail ?? (rHealthy
                  ? 'R, if you have it installed, can also run this method — useful as a second opinion.'
                  : 'R cannot run this method here yet.')}>
            <span className={`dot ${rHealthy ? 'healthy' : engines?.r.found ? 'degraded' : 'unavailable'}`} />R
          </span>
        </div>
      </div>

      {card.why.map((w, i) => (
        <p key={i} className="method-why"><span className="why-mark" aria-hidden>✓</span>{w}</p>
      ))}
      {card.against.map((w, i) => (
        <p key={i} className="method-against"><span className="why-mark" aria-hidden>!</span>{w}</p>
      ))}
      {card.what_can_go_wrong && (
        <p className="method-risk"><strong>What can go wrong.</strong> {card.what_can_go_wrong}</p>
      )}
      {card.diagnostics.length > 0 && (
        <p className="tiny hint">
          Checks it will draw for you:{' '}
          {card.diagnostics.map((d) => d.title).join(', ')}.
        </p>
      )}

      {shownOptions.length > 0 && (
        <details className="method-options" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
          <summary className="tiny">Settings for this method</summary>
          <div className="option-grid">
            {shownOptions.map((o) => (
              <OptionField key={o.name} option={o} value={options[o.name]} onChange={(v) => onOption(o, v)} />
            ))}
          </div>
        </details>
      )}

      {included && pyAvailable && rHealthy && (
        <div className="row engine-choice">
          <label htmlFor={`eng-${card.id}`}>Run on</label>
          <select id={`eng-${card.id}`} value={engine}
                  onChange={(e) => onEngine(e.target.value as 'python' | 'r' | 'both')}>
            <option value="python">Python</option>
            <option value="r">R</option>
            <option value="both">Both, and compare them</option>
          </select>
          <span className="tiny hint">
            Two independent programs implement this method. Running both draws them one under the other
            on the same chart with their ranges of uncertainty — a forest plot — so you can see at a
            glance whether they land in the same place. Where they do not, the two have made different
            small choices, and that is much better to find out here than from a referee.{' '}
            <button className="linky" onClick={() => setExplain('concept.forest_plot')}>
              How to read that chart
            </button>
          </span>
        </div>
      )}

      <div className="method-foot row">
        <button className="btn ghost sm" onClick={() => setExplain(card.explain_key)}>Explain</button>
        {card.references?.length ? (
          <details className="method-references"><summary>Literature · {card.references.length} sources</summary><References references={card.references} compact /></details>
        ) : null}
        <div className="spacer" />
        <button
          className={`btn sm${included ? '' : ' primary'}`}
          onClick={onToggle}
          disabled={card.blocked || (!pyAvailable && !rHealthy)}
          title={
            card.blocked ? 'Go back to the board and fill in the boxes marked “needed” first'
              : !pyAvailable && !rHealthy
                ? (health?.r?.detail ?? 'Neither calculator can run this method yet')
                : rHealthy && pyAvailable
                  ? 'Two programs implement this; the comparison can run it on each and show both'
                  : undefined
          }
        >
          {included ? 'Remove from comparison' : 'Include in comparison'}
        </button>
      </div>
    </article>
  )
}

/** The choices in a method's drop-down arrive as bare storage values, so a menu
 *  offered "ps", "logit_ps", "mahalanobis" or, worse, "t", "s", "x", "dr". The
 *  ones a reader cannot decode are named here; the rest are readable once the
 *  underscores go, and standard acronyms (ATE, HC1) are left as they are written
 *  in every paper the user will meet. Only genuine names are expanded — nothing
 *  is upper-cased by pattern, which is how "n" once became "N". */
const CHOICE_LABELS: Record<string, string> = {
  ps: 'Propensity score',
  logit_ps: 'Propensity score, on the logit scale',
  mahalanobis: 'Mahalanobis distance',
  ipw: 'Inverse-probability weighting',
  ols: 'Ordinary least squares',
  liml: 'Limited-information maximum likelihood',
  '2sls': 'Two-stage least squares',
  t: 'T-learner (one model per arm)',
  s: 'S-learner (one model, treatment as a variable)',
  x: 'X-learner',
  dr: 'DR-learner (doubly robust)',
  ct: 'Causal tree',
  cace: 'Effect on those the instrument moved (CACE)',
  itt: 'Effect of being offered it (intention to treat)',
  att: 'ATT — the people who actually got it',
  ate: 'ATE — everybody',
  atc: 'ATC — the people who did not get it',
  ato: 'ATO — the people who could have gone either way',
  stabilised: 'Stabilised weights',
  gradient_boosting: 'Gradient boosting',
  ik: 'Imbens–Kalyanaraman bandwidth',
  mse: 'Bandwidth that minimises expected error',
  hac: 'Errors allowed to be correlated over time',
  cv: 'Chosen by cross-validation',
  welch_t: 'Welch t-test (unequal spread)',
  rank_sum: 'Rank-sum test',
  diff_means: 'Difference in means',
  'step+ramp': 'A jump and then a change of slope',
  two_sided: 'Two-sided',
  'two-sided': 'Two-sided',
}

function choiceLabel(choice: string): string {
  const known = CHOICE_LABELS[choice]
  if (known) return known
  if (!choice.includes('_')) return choice
  return choice.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())
}

function OptionField({
  option, value, onChange,
}: { option: MethodOption; value: unknown; onChange: (v: unknown) => void }) {
  const v = value === undefined ? option.default : value
  const id = `opt-${option.name}`
  if (option.type === 'bool') {
    return (
      <label className="option-row" htmlFor={id}>
        <input id={id} type="checkbox" checked={!!v} onChange={(e) => onChange(e.target.checked)} />
        <span>
          {option.label}
          {option.help && <span className="tiny hint"> {option.help}</span>}
        </span>
      </label>
    )
  }
  if (option.type === 'select') {
    return (
      <div className="field">
        <label htmlFor={id}>{option.label}</label>
        <select id={id} value={String(v ?? '')} onChange={(e) => onChange(e.target.value)}>
          {(option.choices ?? []).map((c) => <option key={c} value={c}>{choiceLabel(c)}</option>)}
        </select>
        {option.help && <span className="hint">{option.help}</span>}
      </div>
    )
  }
  if (option.type === 'number' || option.type === 'int') {
    return (
      <div className="field">
        <label htmlFor={id}>{option.label}</label>
        <input
          id={id}
          type="number"
          value={v === null || v === undefined ? '' : String(v)}
          min={option.min}
          max={option.max}
          step={option.type === 'int' ? 1 : 'any'}
          onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
        />
        {option.help && <span className="hint">{option.help}</span>}
      </div>
    )
  }
  return (
    <div className="field">
      <label htmlFor={id}>{option.label}</label>
      <input id={id} type="text" value={v === null || v === undefined ? '' : String(v)}
             onChange={(e) => onChange(e.target.value)} />
      {option.help && <span className="hint">{option.help}</span>}
    </div>
  )
}
