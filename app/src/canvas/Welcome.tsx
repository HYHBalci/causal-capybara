/** On launch, not a blank grey window. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { isDesktop, pickFile } from '../desktop'
import { useStore } from '../store'
import { ExternalLink } from '../components/References'
import type { ExampleTile, LearnArticle } from '../types'
import { count } from '../format'
import { useDialogFocus } from '../components/useDialogFocus'

/** A design's name in both registers.
 *
 * Every tile used to be labelled with the engine's own identifier -- "did",
 * "rd", "iv", "its" -- and the filter offered the same seven codes. Nobody who
 * has not taken the course can read those, and nobody who has needs them. The
 * catalogue already carries a plain-language title for each design and the
 * names the literature uses for the same thing, so a tile can say "Policy
 * rolled out over time" with "Difference-in-differences" underneath it: the
 * newcomer only needs the first line, the analyst recognises the second.
 */
interface DesignName {
  title: string
  /** The textbook name, shown as the quiet second line. */
  textbook: string | null
  /** Every name the literature uses, so the search box answers to them too. */
  aliases: string[]
}

type TourStep = NonNullable<LearnArticle['tour']>[number]

/** An example the reader has asked about but not yet opened. */
interface Preview {
  tile: ExampleTile
  steps: TourStep[] | null
  error: string | null
}

export function Welcome() {
  const store = useStore()
  const [examples, setExamples] = useState<ExampleTile[]>([])
  const [examplesError, setExamplesError] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [creating, setCreating] = useState(false)
  const closePreview = useCallback(() => setPreview(null), [])
  const [exampleQuery, setExampleQuery] = useState('')
  const [exampleDesign, setExampleDesign] = useState('all')
  const [datasets, setDatasets] = useState<ExampleTile[]>([])
  const [datasetsError, setDatasetsError] = useState<string | null>(null)
  const [catalogue, setCatalogue] = useState<Record<string, DesignName>>({})
  const [preview, setPreview] = useState<Preview | null>(null)

  useEffect(() => {
    api.examples().then(setExamples).catch((e) => setExamplesError(String(e?.detail ?? e)))
    api.datasets().then(setDatasets).catch((e) => setDatasetsError(String(e?.detail ?? e)))
    // The outline is the one request that knows both registers of every name.
    // If it does not answer, the design titles already in the store carry the
    // screen on their own and only the second line goes missing.
    api.learnOutline()
      .then((rows) => {
        const named: Record<string, DesignName> = {}
        for (const row of rows) {
          if (row.kind !== 'design') continue
          named[row.id] = {
            title: row.title,
            textbook: row.also_called[0] ?? null,
            aliases: row.also_called,
          }
        }
        setCatalogue(named)
      })
      .catch(() => undefined)
  }, [])

  // The gallery above is generated and always available; this one has to be
  // downloaded, so a tile has two states and says which it is in.
  async function downloadDataset(id: string) {
    const ok = await store.fetchDataset(id)
    if (ok) await api.datasets().then(setDatasets).catch(() => undefined)
  }

  const designNames = useMemo(() => {
    const named: Record<string, DesignName> = {}
    for (const design of store.designs) {
      named[design.id] = { title: design.title, textbook: null, aliases: [] }
    }
    return { ...named, ...catalogue }
  }, [store.designs, catalogue])

  /** Never show a bare identifier. If neither source knows this design, the
   *  identifier is all there is, and printing it beats printing nothing. */
  function designName(id: string): DesignName {
    return designNames[id] ?? { title: id, textbook: null, aliases: [] }
  }

  const exampleDesigns = useMemo(
    () => [...new Set(examples.map((e) => e.design))]
      .map((id) => ({ id, ...designName(id) }))
      .sort((a, b) => a.title.localeCompare(b.title)),
    // designName reads designNames, which is what actually changes.
    [examples, designNames],
  )
  // A first-time user landing on a naming field has to know what they are doing
  // before the app has told them anything. These three are the shortest route to
  // a board with real numbers on it.
  const starters = useMemo(
    () => examples.filter((e) => e.difficulty === 'start here').slice(0, 3),
    [examples],
  )

  const shownExamples = useMemo(() => {
    const query = exampleQuery.trim().toLocaleLowerCase()
    return examples.filter((e) => {
      if (exampleDesign !== 'all' && e.design !== exampleDesign) return false
      if (!query) return true
      const design = designName(e.design)
      return [e.title, e.blurb, e.teaches, e.citation, design.title, ...design.aliases]
        .some((value) => value?.toLocaleLowerCase().includes(query))
    })
  }, [examples, exampleDesign, exampleQuery, designNames])

  /** Show what the example is going to teach before opening it.
   *
   * Each example ships a hand-written tour -- the six or seven things to look
   * at, in order -- and until now nothing rendered it, so the first click
   * landed a beginner cold on a half-filled design board. The catalogue serves
   * the same tour without building anything, so reading it costs nothing and
   * commits to nothing. */
  function openPreview(tile: ExampleTile) {
    setPreview({ tile, steps: null, error: null })
    api.learnEntry('example', tile.id)
      .then((article) => {
        setPreview((now) => (now?.tile.id === tile.id ? { ...now, steps: article.tour ?? [] } : now))
      })
      .catch((e) => {
        setPreview((now) => (
          now?.tile.id === tile.id ? { ...now, steps: [], error: String(e?.detail ?? e) } : now
        ))
      })
  }

  async function createProject() {
    if (!name.trim() || creating) return
    setCreating(true)
    try { await store.newProject(name.trim()) } finally { setCreating(false) }
  }

  return (
    <div className="welcome scroll">
      <header className="welcome-head">
        <span className="welcome-eyebrow">A local workspace for causal inference</span>
        <h1>Causal Capybara</h1>
        <p className="welcome-tag">Clear questions. Credible evidence.</p>
        <p className="welcome-lede">
          Explore cause and effect with guided research designs, transparent diagnostics, and
          reproducible analysis. Start with an example, or bring your own data and choose the
          treatment, outcome, and assumptions behind your question.
        </p>
      </header>

      {starters.length > 0 && (
        <section className="welcome-starters" aria-label="Start here">
          <div className="row welcome-starters-head">
            <h2>Start with a guided example</h2>
            <a className="linky tiny" href="#gallery">See all {examples.length} examples ↓</a>
          </div>
          <p className="hint">
            Your question and study design are already set up. Use simulated data with a known
            effect to see how an analysis works, step by step. Each example opens in its own project.
          </p>
          <div className="starter-row">
            {starters.map((e) => {
              const design = designName(e.design)
              return (
                <button key={e.id} className="starter card" onClick={() => openPreview(e)}>
                  <span className="starter-design">{design.title}</span>
                  {design.textbook && <span className="tiny hint">{design.textbook}</span>}
                  <span className="starter-title">{e.title}</span>
                  <span className="starter-teaches tiny">{e.teaches}</span>
                  <span className="starter-go tiny">See what you will do →</span>
                </button>
              )
            })}
          </div>
        </section>
      )}

      <div className="welcome-grid">
        <section className="card welcome-card">
          <h2>Start a project</h2>
          <div className="field">
            <label htmlFor="new-name">Name</label>
            <input
              id="new-name"
              type="text"
              value={name}
              placeholder="minimum wage study"
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') void createProject() }}
            />
            <span className="hint">
              A local folder keeps your data, questions, and saved analyses together.
            </span>
          </div>
          <button className="btn primary" disabled={!name.trim() || creating} onClick={() => void createProject()}>
            {creating ? 'Creating project…' : 'Create project'}
          </button>
          <div className="welcome-sep" />
          <h3>Open an existing project</h3>
          <OpenByPath />
        </section>

        <section className="card welcome-card">
          <h2>Find your research design</h2>
          <p className="hint">
            Browse research designs, methods, and the assumptions they need. Read the catalogue
            without a dataset or a running engine.
          </p>
          <button className="btn" style={{ marginTop: 10 }} onClick={() => store.setView('learn')}>
            Explore the method catalogue
          </button>
        </section>

        <section className="card welcome-card">
          <h2>Recent projects</h2>
          {store.recents.length === 0 && <p className="hint">Your projects will appear here after you open one.</p>}
          <ul className="recent-list">
            {store.recents.map((r) => (
              <li key={r.path}>
                <button className="recent-item" onClick={() => void store.openProject(r.path)}>
                  <span className="recent-name">{r.name}</span>
                  <span className="recent-path tiny hint">{r.path}</span>
                </button>
              </li>
            ))}
          </ul>
          <p className="tiny hint" style={{ marginTop: 10 }}>
            New projects are saved in <span className="mono">{store.workspace}</span>
          </p>
        </section>
      </div>

      <section className="welcome-examples" id="gallery">
        <div className="row welcome-examples-head">
          <div>
            <h2>Explore the example gallery</h2>
            <p className="hint">
              Practice with simulated datasets inspired by published research. Each example
              highlights a different way to identify a causal effect; no original study records are included.
            </p>
          </div>
          {examples.length > 0 && <span className="chip teal">{examples.length} worked examples</span>}
        </div>
        <p className="hint">
          Preview the guided tour, then open a project ready for analysis.
        </p>
        {examplesError && (
          <div className="banner caution">
            <span>The examples are not available in this build ({examplesError}).</span>
          </div>
        )}
        {examples.length > 0 && (
          <div className="example-tools" role="search" aria-label="Filter example datasets">
            <label className="sr-only" htmlFor="example-search">Search example datasets</label>
            <input
              id="example-search"
              type="search"
              value={exampleQuery}
              placeholder="Search by subject, method or study…"
              onChange={(e) => setExampleQuery(e.target.value)}
            />
            <label className="sr-only" htmlFor="example-design">Show only one kind of study</label>
            <select id="example-design" value={exampleDesign} onChange={(e) => setExampleDesign(e.target.value)}>
              <option value="all">Every kind of study</option>
              {exampleDesigns.map((design) => (
                <option key={design.id} value={design.id}>
                  {design.textbook ? `${design.title} (${design.textbook.toLocaleLowerCase()})` : design.title}
                </option>
              ))}
            </select>
            <span className="tiny hint" aria-live="polite">
              Showing {shownExamples.length} of {examples.length}
            </span>
          </div>
        )}
        <div className="example-grid">
          {shownExamples.map((e) => {
            const design = designName(e.design)
            return (
              <button
                key={e.id}
                className="example-tile card"
                onClick={() => openPreview(e)}
                aria-label={`${e.title}: see what this example teaches before opening it`}
              >
                <div className="row example-kicker">
                  <span className="col" style={{ gap: 1 }}>
                    <span className="example-design">{design.title}</span>
                    {design.textbook && <span className="tiny hint">{design.textbook}</span>}
                  </span>
                  <span className="chip moss">Simulated data</span>
                </div>
                <div className="example-title">{e.title}</div>
                <p className="example-blurb">{e.blurb}</p>
                {e.teaches && <p className="example-teaches">Teaches: {e.teaches}</p>}
                {e.citation && <p className="example-citation">Inspired by {e.citation.replace(/^After\s+/i, '')}</p>}
                <div className="row tiny hint example-meta">
                  <span>{e.n ? `${count(e.n)} rows` : 'Built when you open it'}</span>
                  {e.difficulty ? <span>· {e.difficulty}</span> : null}
                  <span className="spacer" />
                  <span className="example-open">See what you will do →</span>
                </div>
              </button>
            )
          })}
        </div>
        {examples.length > 0 && shownExamples.length === 0 && (
          <div className="empty-state example-empty">
            <h3>No examples match</h3>
            <p className="hint">Try a different search, or put the filter back to every kind of study.</p>
            <button className="btn" onClick={() => { setExampleQuery(''); setExampleDesign('all') }}>
              Clear filters
            </button>
          </div>
        )}
      </section>

      <section className="welcome-examples">
        <div className="row welcome-examples-head">
          <h2>The real studies</h2>
          {datasets.length > 0 && (
            <span className="chip">
              {datasets.filter((d) => d.cached).length} of {datasets.length} downloaded
            </span>
          )}
        </div>
        <p className="hint">
          Work with published datasets, downloaded only when you request them. Files are checked
          for integrity and stored locally. Unlike the simulated examples, these studies have no
          known causal effect to check your result against.
        </p>
        {datasetsError && (
          <div className="empty-state example-empty">
            <span>Study data is not available in this build ({datasetsError}).</span>
          </div>
        )}
        <div className="example-grid">
          {datasets.map((d) => {
            const design = designName(d.design)
            return (
              <div key={d.id} className="example-tile card study-tile">
                <div className="row example-kicker">
                  <span className="col" style={{ gap: 1 }}>
                    <span className="example-design">{design.title}</span>
                    {design.textbook && <span className="tiny hint">{design.textbook}</span>}
                  </span>
                  {d.cached
                    ? <span className="chip teal">Study data · on this machine</span>
                    : <span className="chip">Study data · not downloaded</span>}
                </div>
                <div className="example-title">{d.title}</div>
                <p className="example-blurb">{d.blurb}</p>
                {d.teaches && <p className="example-teaches">Teaches: {d.teaches}</p>}
                {d.citation && (
                  <p className="example-citation">
                    {d.citation_url
                      ? <ExternalLink href={d.citation_url}>{d.citation}</ExternalLink>
                      : d.citation}
                  </p>
                )}
                {d.caveat && <p className="example-teaches">Note: {d.caveat}</p>}
                <div className="row tiny hint example-meta">
                  <span>{d.n ? `${count(d.n)} rows` : ''}</span>
                  {d.rights ? <span>· {d.rights}</span> : null}
                </div>
                <div className="row tiny example-meta">
                  <span className="spacer" />
                  {d.cached ? (
                    <button className="btn" onClick={() => void store.openDataset(d.id)}>
                      Open project →
                    </button>
                  ) : (
                    <button className="btn" onClick={() => void downloadDataset(d.id)}>
                      Download from publisher
                    </button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </section>

      <footer className="welcome-foot hint tiny">
        Your analyses run locally. No usage tracking. Study downloads, package installation, and
        publication links use the internet when you request them.
      </footer>

      {preview && (
        <ExampleTour
          preview={preview}
          design={designName(preview.tile.design)}
          onClose={closePreview}
          onOpen={() => { setPreview(null); void store.openExample(preview.tile.id) }}
          onLearn={() => { setPreview(null); store.setView('learn') }}
        />
      )}
    </div>
  )
}

/** The guided tour, at last on a screen.
 *
 * It reads before the project is built rather than after, because the point of
 * a tour is to say what you are about to do. The example is only created when
 * the reader says yes, so closing this costs them nothing on disk.
 */
function ExampleTour(
  { preview, design, onClose, onOpen, onLearn }: {
    preview: Preview
    design: DesignName
    onClose: () => void
    onOpen: () => void
    onLearn: () => void
  },
) {
  const { tile, steps, error } = preview

  const dialogRef = useRef<HTMLDivElement>(null)
  useDialogFocus(dialogRef, true, onClose)

  return (
    <div className="palette-scrim" onClick={onClose}>
      <div
        className="card pad col"
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={`What you will do in ${tile.title}`}
        onClick={(e) => e.stopPropagation()}
        style={{ width: 'min(620px, 92vw)', maxHeight: '78vh' }}
      >
        <div className="row">
          <span className="col" style={{ gap: 1 }}>
            <span className="example-design">{design.title}</span>
            {design.textbook && <span className="tiny hint">{design.textbook}</span>}
          </span>
          <span className="spacer" />
          <button className="btn sm" onClick={onClose}>Close</button>
        </div>
        <h2 style={{ margin: '8px 0 0', fontSize: 17 }}>{tile.title}</h2>
        <p className="hint" style={{ margin: '4px 0 0' }}>{tile.blurb}</p>
        {tile.citation_url && <p className="example-citation"><ExternalLink href={tile.citation_url}>{tile.citation ?? 'Read the source study'}</ExternalLink></p>}

        <div className="scroll grow" style={{ marginTop: 12 }}>
          <div className="panel-title">What you will do, in order</div>
          {steps === null && <p className="hint">Fetching the tour…</p>}
          {steps !== null && steps.length === 0 && (
            <p className="hint">
              {error
                ? 'The step-by-step tour could not be loaded, but the example itself still opens.'
                : 'This example does not come with a tour yet. It still opens ready to run.'}
            </p>
          )}
          {steps !== null && steps.length > 0 && (
            <ol style={{ margin: '8px 0 0', paddingLeft: 20, display: 'grid', gap: 12 }}>
              {steps.map((step, i) => (
                <li key={i}>
                  <div style={{ fontWeight: 620 }}>{String(step.title ?? '')}</div>
                  <p className="hint" style={{ margin: '2px 0 0', lineHeight: 1.45 }}>
                    {String(step.body ?? '')}
                  </p>
                </li>
              ))}
            </ol>
          )}
        </div>

        <div className="row" style={{ gap: 8, marginTop: 14 }}>
          <button className="btn primary" onClick={onOpen}>Open this example</button>
          <button className="linky tiny" onClick={onLearn}>Read about this design first</button>
          <span className="spacer" />
          <span className="tiny hint">Opens a separate project.</span>
        </div>
      </div>
    </div>
  )
}

/** Opening a project that already exists.
 *
 * Typing an absolute path was the only way in, which is a full stop for anyone
 * who has never used a terminal -- and the desktop shell has been able to show
 * the operating system's own file dialog the whole time. The typed field stays
 * as the second way in, for a path pasted from somewhere else and for the
 * browser tab used during development, where there is no dialog to show.
 */
function OpenByPath() {
  const openProject = useStore((s) => s.openProject)
  const toast = useStore((s) => s.toast)
  const [path, setPath] = useState('')

  const browse = async () => {
    const picked = await pickFile([{ name: 'Causal Capybara project', extensions: ['yaml'] }])
    if (picked) {
      // A project is a folder, and folders cannot be picked from a file dialog,
      // so the reader picks a file inside one. The engine opens either the
      // folder or its project.yaml, so anything else is handed over as the
      // folder it sits in.
      const isProjectFile = /(^|[\\/])project\.yaml$/i.test(picked)
      void openProject(isProjectFile ? picked : picked.replace(/[\\/][^\\/]*$/, ''))
    } else if (!isDesktop()) {
      toast('info', 'Choosing a folder needs the installed app.',
        'In a browser tab, paste the full path of the project folder instead.')
    }
  }

  return (
    <>
      <button className="btn" onClick={() => void browse()}>Choose a project…</button>
      <p className="hint tiny" style={{ margin: '6px 0 8px' }}>
        A project is a folder whose name ends in <span className="mono">.capy</span>. Open it and
        choose the file called <span className="mono">project.yaml</span> inside.
      </p>
      <div className="row" style={{ gap: 6 }}>
        <input
          className="grow"
          type="text"
          aria-label="Project folder path"
          placeholder="or paste the folder here"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && path.trim()) void openProject(path.trim()) }}
        />
        <button className="btn" disabled={!path.trim()} onClick={() => void openProject(path.trim())}>
          Open
        </button>
      </div>
    </>
  )
}
