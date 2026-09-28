/** The catalogue: everything this app knows, readable on its own terms.
 *
 * Until now the only route to a method card ran through a project, a dataset
 * and a chosen design, and the one advertised shortcut -- a design card reading
 * "I already know the method" -- opened a screen that needs a question to
 * recommend against, so it sat on a spinner for ever. Somebody who simply wants
 * to know what a Love plot is, or which estimator to read about before they
 * have any data, had nowhere to go.
 *
 * So: nine sections, 279 articles, a search box that answers to the names the
 * literature uses as well as the ones on the cards, and no project required.
 * Where an article is thin, it says which part is missing rather than
 * rendering an empty heading -- the catalogue is honest about its own gaps for
 * the same reason the estimates are.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import type { LearnArticle, LearnKind, LearnRef, LearnRow } from '../types'
import { ExternalLink, References } from '../components/References'

const SECTIONS: { kind: LearnKind; title: string; blurb: string }[] = [
  { kind: 'design', title: 'Research designs', blurb: 'How the treatment came to be assigned. Everything else follows from this.' },
  { kind: 'method', title: 'Methods', blurb: 'The estimators themselves: what each does, and where each breaks.' },
  { kind: 'probe', title: 'Robustness probes', blurb: 'Ways of asking whether an answer survives being poked.' },
  { kind: 'estimand', title: 'What the number means', blurb: 'Which effect, and for whom. Two correct methods can answer different questions.' },
  { kind: 'assumption', title: 'Assumptions', blurb: 'The claims a result rests on, and whether anything can test them.' },
  { kind: 'diagnostic', title: 'Checks and plots', blurb: 'What the app checks for you, and what each one is looking at.' },
  { kind: 'concept', title: 'Ideas worth knowing', blurb: 'The background the rest of the catalogue leans on.' },
  { kind: 'example', title: 'Worked examples', blurb: 'Simulated studies with a built-in answer, so you can check what estimators recover.' },
  { kind: 'study', title: 'Published study data', blurb: 'Real datasets from the literature. No built-in answer — that is the point.' },
]

/** The catalogue as it was when the app was built.
 *
 * Loaded on demand, and only when the engine cannot answer: it is about 1.8 MB
 * of JSON, which has no business in the startup path of a window that usually
 * has a live engine to ask. Written by tools/build_catalogue.py.
 */
interface FrozenCatalogue {
  outline: LearnRow[]
  articles: Record<string, LearnArticle>
}

let frozen: Promise<FrozenCatalogue> | null = null

function bundledCatalogue(): Promise<FrozenCatalogue> {
  if (!frozen) {
    frozen = import('../generated/catalogue.json').then(
      (m) => (m.default ?? m) as unknown as FrozenCatalogue,
    )
  }
  return frozen
}

/** Offline reading must never wait for an engine request to time out. */
export async function loadLearnOutline(engineDown: boolean): Promise<{ rows: LearnRow[]; offline: boolean }> {
  if (!engineDown) {
    try { return { rows: await api.learnOutline(), offline: false } } catch { /* use the installed copy */ }
  }
  return { rows: (await bundledCatalogue()).outline, offline: true }
}

export async function loadLearnArticle(kind: string, id: string, engineDown: boolean): Promise<LearnArticle> {
  if (!engineDown) {
    try { return await api.learnEntry(kind, id) } catch { /* use the installed copy */ }
  }
  const found = (await bundledCatalogue()).articles[`${kind}:${id}`]
  if (!found) throw new Error(`The catalogue has no ${kind} called ‘${id}’.`)
  return found
}

export function Learn() {
  const [rows, setRows] = useState<LearnRow[]>([])
  const [offline, setOffline] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<{ kind: string; id: string } | null>(null)
  const [openKinds, setOpenKinds] = useState<Set<string>>(new Set(['design']))
  const [history, setHistory] = useState<{ kind: string; id: string }[]>([])
  const articleScroll = useRef<HTMLDivElement>(null)
  const pick = (kind: string, id: string) => {
    if (selected?.kind === kind && selected?.id === id) return
    if (selected) setHistory((items) => [...items, selected].slice(-60))
    setSelected({ kind, id })
  }
  useEffect(() => { articleScroll.current?.scrollTo({ top: 0 }) }, [selected?.kind, selected?.id])
  const engineDown = useStore((s) => s.engineDown)

  useEffect(() => {
    let live = true
    // When the engine is down the bundled copy is the primary source, not a
    // fallback behind a request that could take two minutes to time out.
    loadLearnOutline(engineDown).then(
      ({ rows: outline, offline: bundled }) => {
        if (live) { setRows(outline); setOffline(bundled); setError(null) }
      },
      (err) => { if (live) setError(String((err as Error)?.message ?? err)) },
    )
    return () => { live = false }
  }, [engineDown])

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return null
    return rows.filter((r) =>
      r.title.toLowerCase().includes(q)
      || r.id.toLowerCase().includes(q)
      || (r.one_liner ?? '').toLowerCase().includes(q)
      || r.also_called.some((a) => a.toLowerCase().includes(q))
      || r.tags.some((t) => t.toLowerCase().includes(q)))
  }, [rows, query])

  const byKind = useMemo(() => {
    const out = new Map<string, LearnRow[]>()
    for (const r of rows) {
      const list = out.get(r.kind) ?? []
      list.push(r)
      out.set(r.kind, list)
    }
    return out
  }, [rows])

  const toggle = (kind: string) =>
    setOpenKinds((prev) => {
      const next = new Set(prev)
      if (next.has(kind)) next.delete(kind)
      else next.add(kind)
      return next
    })

  if (error) {
    return (
      <div className="canvas-pad">
        <div className="banner caution">
          <span>The catalogue could not be opened. {error}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="learn">
      <nav className="learn-rail" aria-label="Catalogue">
        <div className="learn-search">
          <input
            type="search"
            className="learn-search-input"
            placeholder="Search — try “difference-in-differences” or “overlap”"
            aria-label="Search the catalogue"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <p className="tiny hint learn-count">
            {matches
              ? `${matches.length} of ${rows.length} entries`
              : `${rows.length} entries. The names on the cards are plain English; searching the
                 textbook name finds them too.`}
          </p>
          {offline && (
            <p className="tiny hint learn-count">
              Reading the catalogue bundled with the app. It is available offline and reflects the
              content included in this release.
            </p>
          )}
        </div>

        <div className="learn-rail-body scroll">
          {matches ? (
            <ul className="learn-list">
              {matches.map((r) => (
                <LearnRowButton
                  key={`${r.kind}:${r.id}`}
                  row={r}
                  showKind
                  selected={selected?.kind === r.kind && selected?.id === r.id}
                  onPick={() => pick(r.kind, r.id)}
                />
              ))}
              {matches.length === 0 && (
                <li className="pad hint tiny">
                  Nothing matches “{query}”. The catalogue covers designs, methods, estimands,
                  assumptions, diagnostics and the worked examples.
                </li>
              )}
            </ul>
          ) : (
            SECTIONS.map((s) => {
              const list = byKind.get(s.kind) ?? []
              const open = openKinds.has(s.kind)
              return (
                <section key={s.kind} className="learn-section">
                  <button
                    className="learn-section-head"
                    onClick={() => toggle(s.kind)}
                    aria-expanded={open}
                  >
                    <span className="learn-caret" aria-hidden>{open ? '▾' : '▸'}</span>
                    <span className="grow">{s.title}</span>
                    <span className="tiny hint">{list.length}</span>
                  </button>
                  {open && (
                    <>
                      <p className="tiny hint learn-section-blurb">{s.blurb}</p>
                      <ul className="learn-list">
                        {list.map((r) => (
                          <LearnRowButton
                            key={`${r.kind}:${r.id}`}
                            row={r}
                            selected={selected?.kind === r.kind && selected?.id === r.id}
                            onPick={() => pick(r.kind, r.id)}
                          />
                        ))}
                      </ul>
                    </>
                  )}
                </section>
              )
            })
          )}
        </div>
      </nav>

      <div className="learn-article scroll" ref={articleScroll}>
        <div className="article-toolbar">
          <button className="btn ghost sm" disabled={!selected} onClick={() => {
            setSelected(history[history.length - 1] ?? null)
            setHistory((items) => items.slice(0, -1))
          }}>← Previous article</button>
          <button className="btn ghost sm" onClick={() => { setSelected(null); setHistory([]) }}>Catalogue home</button>
          <span className="spacer" />
          <button className="btn ghost sm" onClick={() => useStore.getState().setView('literature')}>Literature library ↗</button>
        </div>
        {selected
          ? <Article key={`${selected.kind}:${selected.id}`} kind={selected.kind} id={selected.id} bundledOnly={engineDown || offline} onGo={pick} />
          : <LearnIntro rows={rows} onPick={pick} />}
      </div>
    </div>
  )
}

function LearnRowButton({ row, selected, showKind, onPick }: {
  row: LearnRow
  selected: boolean
  showKind?: boolean
  onPick: () => void
}) {
  return (
    <li>
      <button
        className={`learn-item${selected ? ' on' : ''}`}
        onClick={onPick}
        aria-current={selected ? 'true' : undefined}
      >
        <span className="learn-item-title">{row.title}</span>
        {row.one_liner && <span className="learn-item-sub">{row.one_liner}</span>}
        {showKind && <span className="learn-item-kind">{kindLabel(row.kind)}</span>}
      </button>
    </li>
  )
}

function LearnIntro({ rows, onPick }: {
  rows: LearnRow[]
  onPick: (kind: string, id: string) => void
}) {
  const starts: [LearnKind, string][] = [
    ['concept', 'concept.confounding'],
    ['concept', 'concept.collider'],
    ['design', 'observational'],
    ['design', 'did'],
    ['estimand', 'ATT'],
    ['diagnostic', 'love'],
  ]
  const known = new Set(rows.map((r) => `${r.kind}:${r.id}`))
  const byId = new Map(rows.map((r) => [`${r.kind}:${r.id}`, r]))
  return (
    <div className="learn-intro">
      <h1>The catalogue</h1>
      <p className="learn-lede">
        Every design, method, estimand, assumption and check this app can run, written to be read
        rather than looked up. You do not need a project open, and nothing here changes anything.
      </p>
      <p className="learn-lede">
        If you are new to this: start with what confounding is, then read the design that matches how
        your treatment was decided. The design is the part that does the work — the method is a
        detail inside it.
      </p>
      <div className="learn-starts">
        {starts.filter(([k, i]) => known.has(`${k}:${i}`)).map(([k, i]) => {
          const row = byId.get(`${k}:${i}`)!
          return (
            <button key={`${k}:${i}`} className="card pad learn-start" onClick={() => onPick(k, i)}>
              <span className="learn-item-kind">{kindLabel(k)}</span>
              <strong>{row.title}</strong>
              {row.one_liner && <span className="tiny hint">{row.one_liner}</span>}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function Article({ kind, id, bundledOnly, onGo }: {
  kind: string
  id: string
  bundledOnly: boolean
  onGo: (kind: string, id: string) => void
}) {
  const [article, setArticle] = useState<LearnArticle | null>(null)
  const [error, setError] = useState<string | null>(null)
  const top = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let live = true
    setArticle(null)
    setError(null)
    loadLearnArticle(kind, id, bundledOnly).then(
      (loaded) => { if (live) setArticle(loaded) },
      (err) => { if (live) setError(String((err as Error)?.message ?? err)) },
    )
    // A new article starts at its own beginning, not half way down the last one.
    top.current?.scrollTo?.({ top: 0 })
    return () => { live = false }
  }, [kind, id, bundledOnly])

  if (error) return <div className="pad"><div className="banner caution"><span>{error}</span></div></div>
  if (!article) {
    return <div className="pad row" style={{ gap: 8 }}><span className="spinner" /> Opening…</div>
  }

  const a = article
  return (
    <article className="learn-body" ref={top}>
      <header className="learn-head">
        <span className="learn-item-kind">{kindLabel(a.kind)}</span>
        <h1>{a.title}</h1>
        {a.one_liner && <p className="learn-lede">{a.one_liner}</p>}
        {a.also_called?.length ? (
          <p className="tiny hint">
            Also called {a.also_called.join(', ')}.
          </p>
        ) : null}
        {a.status_label && <span className={`chip ${statusTone(a.status)}`}>{a.status_label}</span>}
      </header>

      <Prose title="What it is" body={a.plain_language} />
      <Prose title="In more depth" body={a.in_more_depth} />
      <Prose title="Use it when" body={a.when_to_use} />
      <Prose title="Do not use it when" body={a.when_not_to_use} />
      <Prose title="How the board reads" body={a.how_the_board_reads} />
      <Prose title="The question it answers" body={a.question} />
      <Prose title="What it teaches" body={a.teaches} />

      {a.worry_when && (
        <div className="banner caution">
          <span><strong>What would worry me.</strong> {a.worry_when}</span>
        </div>
      )}
      {a.common_mistake && (
        <div className="banner caution">
          <span><strong>Common mistake.</strong> {a.common_mistake}</span>
        </div>
      )}
      {a.caveat && (
        <div className="banner info"><span>{a.caveat}</span></div>
      )}

      <Refs title="It rests on" refs={a.what_it_assumes} onGo={onGo} withExplains />
      <Refs title="Designs it belongs to" refs={a.designs} onGo={onGo} />
      <Refs title="What it estimates" refs={a.estimands} onGo={onGo} />
      <Refs title="Methods for it" refs={a.methods} onGo={onGo} />
      <Refs title="Checks it produces" refs={a.diagnostics} onGo={onGo} />
      <Refs title="What produces it" refs={a.produced_by} onGo={onGo} />
      <Refs title="What it speaks to" refs={a.speaks_to} onGo={onGo} />
      <Refs title="How it is checked" refs={a.checks} onGo={onGo} />
      <Refs title="Probes worth running" refs={a.probes} onGo={onGo} />
      <Refs title="Methods to try on it" refs={a.suggested_methods} onGo={onGo} />

      {a.assumption_notes?.length ? (
        <section className="learn-block">
          <h2>Notes on the assumptions</h2>
          <ul className="explain-list">{a.assumption_notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
        </section>
      ) : null}

      {(a.roles_required?.length || a.roles_optional?.length) ? (
        <section className="learn-block">
          <h2>What it needs from your data</h2>
          {a.roles_required?.length ? (
            <p className="learn-roles">
              <strong>Required.</strong> {a.roles_required.map(humaniseRole).join(', ')}.
            </p>
          ) : null}
          {a.roles_optional?.length ? (
            <p className="learn-roles">
              <strong>Optional.</strong> {a.roles_optional.map(humaniseRole).join(', ')}.
            </p>
          ) : null}
          {a.roles_forbidden?.length ? (
            <p className="learn-roles">
              <strong>Never adjust for.</strong> {a.roles_forbidden.map(humaniseRole).join(', ')}.
            </p>
          ) : null}
        </section>
      ) : null}

      {a.availability && (
        <section className="learn-block">
          <h2>Where it runs</h2>
          <p className="learn-roles">
            {a.availability.python ? 'Runs on the Python engine.' : 'No Python version of this yet.'}
            {' '}
            {a.availability.r ? 'Runs on the R engine.' : 'Not implemented on the R engine.'}
          </p>
          {a.availability.note && <p className="tiny hint">{a.availability.note}</p>}
        </section>
      )}

      {(a.citation || a.provenance || a.rows) ? (
        <section className="learn-block">
          <h2>Where the data comes from</h2>
          {a.rows ? <p className="learn-roles"><strong>Rows in the file.</strong> {a.rows}</p> : null}
          {a.provenance && <p className="learn-roles">{a.provenance}</p>}
          {a.citation_url ? <p><ExternalLink href={a.citation_url}>{a.citation ?? 'Read the original study'}</ExternalLink></p>
            : a.citation && <p className="tiny">{a.citation}</p>}
        </section>
      ) : null}

      {a.references?.length ? (
        <section className="learn-block">
          <h2>Read the originals</h2>
          <References references={a.references} />
        </section>
      ) : null}

      <Refs title="See also" refs={a.see_also} onGo={onGo} />

    </article>
  )
}

function Prose({ title, body }: { title: string; body?: string | null }) {
  if (!body) return null
  return (
    <section className="learn-block">
      <h2>{title}</h2>
      {String(body).split(/\n{2,}/).map((p, i) => <p key={i} className="learn-para">{p}</p>)}
    </section>
  )
}

function Refs({ title, refs, onGo, withExplains }: {
  title: string
  refs?: LearnRef[]
  onGo: (kind: string, id: string) => void
  withExplains?: boolean
}) {
  if (!refs?.length) return null
  return (
    <section className="learn-block">
      <h2>{title}</h2>
      <ul className="learn-refs">
        {refs.map((r) => (
          <li key={`${r.kind}:${r.id}`}>
            {r.in_catalogue === false ? (
              <span className="learn-ref-flat">
                <strong>{r.title}</strong>
                {r.one_liner && <span className="tiny hint"> {r.one_liner}</span>}
              </span>
            ) : (
              <button className="learn-ref" onClick={() => onGo(r.kind, r.id)}>
                <strong>{r.title}</strong>
                {r.acronym && r.acronym !== r.title && <span className="tiny hint"> ({r.acronym})</span>}
                {r.one_liner && <span className="learn-ref-sub">{r.one_liner}</span>}
                {withExplains && r.explains && <span className="learn-ref-sub">{r.explains}</span>}
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}

function kindLabel(kind: string): string {
  const labels: Record<string, string> = {
    design: 'Design', method: 'Method', probe: 'Probe', estimand: 'Estimand',
    assumption: 'Assumption', diagnostic: 'Check', concept: 'Idea',
    example: 'Worked example', study: 'Published study',
  }
  return labels[kind] ?? kind
}

function statusTone(status?: string | null): string {
  if (status === 'recommended') return 'moss'
  if (status === 'disrecommended') return 'ochre'
  return ''
}

/** Role names are stored as identifiers; nobody should have to read one. */
function humaniseRole(role: string): string {
  const labels: Record<string, string> = {
    treatment: 'a treatment', outcome: 'an outcome', unit: 'a unit identifier',
    time: 'a time variable', confounders: 'measured confounders', instruments: 'an instrument',
    running: 'a running variable', cutoff: 'a cutoff', mediator: 'a mediator',
    treated_unit: 'the treated unit', event_time: 'an intervention time',
    cluster: 'a clustering variable', donor_pool: 'a donor pool', weight: 'a survey weight',
    strata: 'blocks or strata', control_series: 'a control series',
    effect_modifiers: 'effect modifiers', forbidden: 'anything measured after treatment',
  }
  return labels[role] ?? role.replace(/_/g, ' ')
}
