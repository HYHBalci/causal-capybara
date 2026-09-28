/** The graph editor.
 *
 * Headless schema first, canvas second. Drag variables from the project list
 * onto the canvas, draw directed edges, draw dashed bidirected edges for
 * unmeasured confounding. Live identification shows adjustment SETS, not "the"
 * set. Descendants of the treatment glow red when you try to adjust for them.
 *
 * The caption on the canvas is not decoration: this graph is an assumption you
 * are making.
 */

import {
  useCallback, useEffect, useRef, useState,
  type KeyboardEvent as ReactKeyboardEvent, type MouseEvent as ReactMouseEvent,
} from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { useSplitPane } from '../components/SplitPane'

interface Node {
  id: string
  variable?: string | null
  label?: string | null
  role?: string
  latent?: boolean
  x: number
  y: number
}
interface Edge { from: string; to: string; kind?: 'directed' | 'bidirected'; note?: string | null }
interface Dag { schema: 'capy.dag'; version: 1; id: string; name?: string; nodes: Node[]; edges: Edge[]; caption?: string; selected_adjustment_set?: string[] | null }

const W = 1000
const H = 560
const RX = 46
const RY = 22

/** Where the camera is over the graph.
 *
 * The graph is drawn in a fixed 1000x560 space that the browser then squeezes
 * into whatever the pane happens to be, which on a laptop renders a 13px label
 * at about five. This transform sits between the two: `k` magnifies, `x` and
 * `y` slide, and the identity below is exactly the old fit-the-pane picture.
 */
interface Viewport { x: number; y: number; k: number }
const IDENTITY: Viewport = { x: 0, y: 0, k: 1 }
const MIN_K = 0.4
const MAX_K = 6
// Fit stops well short of the manual ceiling: a graph of one or two nodes
// would otherwise fill the pane with a single enormous ellipse.
const FIT_K = 2.2
const STEP = 1.3

/** Client pixels to the graph's own coordinates, before the camera moves it.
 *
 * Asking the browser for the element's matrix rather than doing the arithmetic
 * by hand is what keeps drops landing under the cursor: the svg is letterboxed
 * inside its pane whenever the pane's shape differs from 1000x560, and a
 * width-only ratio quietly ignores that.
 */
function toRoot(el: SVGSVGElement | null, clientX: number, clientY: number): { x: number; y: number } {
  const ctm = el?.getScreenCTM()
  if (!ctm) return { x: W / 2, y: H / 2 }
  const p = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse())
  return { x: p.x, y: p.y }
}

export function DagEditor() {
  const project = useStore((s) => s.project)
  const spec = useStore((s) => s.spec)
  const columns = useStore((s) => s.columns)
  const setRole = useStore((s) => s.setRole)
  const toast = useStore((s) => s.toast)
  const selectedObjectId = useStore((s) => s.selectedObjectId)
  const selectedGraphId = project?.objects?.some((object) => object.id === selectedObjectId && object.type === 'dag')
    ? selectedObjectId : spec?.dag_id ?? null
  const [dag, setDag] = useState<Dag | null>(null)
  const [ident, setIdent] = useState<Record<string, unknown> | null>(null)
  const [identifying, setIdentifying] = useState(false)
  const [identError, setIdentError] = useState<string | null>(null)
  const [identifyRetry, setIdentifyRetry] = useState(0)
  const [loadRetry, setLoadRetry] = useState(0)
  const [savedGraph, setSavedGraph] = useState<string | null>(null)
  const [savingGraph, setSavingGraph] = useState(false)
  const graphContext = useRef('')
  graphContext.current = JSON.stringify([project?.id, spec?.id, selectedGraphId])
  const graphDirty = !!dag && JSON.stringify(dag) !== savedGraph
  const [drag, setDrag] = useState<{ id: string; dx: number; dy: number } | null>(null)
  const [linkFrom, setLinkFrom] = useState<string | null>(null)
  const [linkKind, setLinkKind] = useState<'directed' | 'bidirected'>('directed')
  const [selected, setSelected] = useState<string | null>(null)
  const [unavailable, setUnavailable] = useState<string | null>(null)
  const [viewport, setViewport] = useState<Viewport>(IDENTITY)
  const [panning, setPanning] = useState(false)
  const svg = useRef<SVGSVGElement>(null)
  const canvas = useRef<HTMLDivElement>(null)
  const split = useSplitPane('split.dag', 860, 360)
  // Where the camera is headed. A wheel notch or a keypress that arrives while
  // the previous one is still gliding has to compound on the destination, not
  // on whatever half-way frame is on screen, or the steps come out uneven.
  const target = useRef<Viewport>(IDENTITY)
  const glide = useRef<number | null>(null)
  const panFrom = useRef<{ rx: number; ry: number; x: number; y: number } | null>(null)
  const fitted = useRef(false)
  const ready = dag !== null

  // An animation frame and a native wheel listener both need the camera as it
  // is right now, and neither can be re-created sixty times a second just to
  // close over fresh state, so every move goes through here.
  const viewportRef = useRef<Viewport>(IDENTITY)
  const setCamera = useCallback((v: Viewport) => { viewportRef.current = v; setViewport(v) }, [])

  /* Move the camera, gently. Someone who has asked their system for less
     motion gets the destination immediately: the ease is there to show which
     way the graph went, and for them it is only a lurch. */
  const moveTo = useCallback((to: Viewport, animate = true) => {
    if (glide.current !== null) cancelAnimationFrame(glide.current)
    target.current = to
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
    if (!animate || reduced) {
      glide.current = null
      setCamera(to)
      return
    }
    const from = viewportRef.current
    const started = performance.now()
    const frame = (now: number) => {
      const t = Math.min(1, (now - started) / 180)
      const e = 1 - (1 - t) ** 3
      setCamera({
        x: from.x + (to.x - from.x) * e,
        y: from.y + (to.y - from.y) * e,
        k: from.k + (to.k - from.k) * e,
      })
      glide.current = t < 1 ? requestAnimationFrame(frame) : null
    }
    glide.current = requestAnimationFrame(frame)
  }, [setCamera])

  /* Zoom around a fixed point, so the wheel magnifies whatever the pointer is
     over instead of the middle of the pane. */
  const zoomAbout = useCallback((factor: number, rx: number, ry: number, animate = true) => {
    const cur = target.current
    const k = Math.min(MAX_K, Math.max(MIN_K, cur.k * factor))
    if (Math.abs(k - cur.k) < 1e-4) return
    const ratio = k / cur.k
    moveTo({ k, x: rx - (rx - cur.x) * ratio, y: ry - (ry - cur.y) * ratio }, animate)
  }, [moveTo])

  /* Put the whole graph in the pane, as large as it will go. */
  const fitTo = useCallback((nodes: Node[], animate = true) => {
    if (!nodes.length) { moveTo(IDENTITY, animate); return }
    const pad = 30
    const x0 = Math.min(...nodes.map((n) => n.x)) - RX - pad
    const x1 = Math.max(...nodes.map((n) => n.x)) + RX + pad
    const y0 = Math.min(...nodes.map((n) => n.y)) - RY - pad
    const y1 = Math.max(...nodes.map((n) => n.y)) + RY + pad
    const k = Math.min(FIT_K, Math.max(MIN_K, Math.min(W / (x1 - x0), H / (y1 - y0))))
    moveTo({ k, x: (W - (x0 + x1) * k) / 2, y: (H - (y0 + y1) * k) / 2 }, animate)
  }, [moveTo])

  /* React registers wheel listeners passively, so preventDefault() from an
     onWheel prop is ignored and the gesture would scroll the window or fire
     the browser's own zoom. A native listener is the only way to claim it. */
  useEffect(() => {
    const el = canvas.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY
      const at = toRoot(svg.current, e.clientX, e.clientY)
      zoomAbout(Math.exp(-dy * 0.0018), at.x, at.y, false)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [zoomAbout, ready])

  // A seeded graph occupies a corner of the 1000x560 space, so the first thing
  // to show is that corner filling the pane -- not a fifth-size sketch. Once
  // only: after this the camera belongs to the user.
  useEffect(() => {
    if (!dag || fitted.current) return
    fitted.current = true
    fitTo(dag.nodes, false)
  }, [dag, fitTo])

  useEffect(() => () => { if (glide.current !== null) cancelAnimationFrame(glide.current) }, [])

  // A saved graph is an object, not a shortcut to a new graph from the roles.
  // Reset on question changes and reject replies from a previous selection.
  useEffect(() => {
    let live = true
    setDag(null)
    setIdent(null)
    setIdentError(null)
    setUnavailable(null)
    setSavedGraph(null)
    setSavingGraph(false)
    setSelected(null)
    setLinkFrom(null)
    fitted.current = false
    moveTo(IDENTITY, false)
    if (!project || !spec) return
    const request = selectedGraphId
      ? api.dag(project.id, selectedGraphId)
      : api.dagFromRoles(spec.roles)
    request.then((d) => {
      if (!live) return
      const loaded = normalise(d as unknown as Dag)
      setDag(loaded)
      setSavedGraph(selectedGraphId ? JSON.stringify(loaded) : null)
    })
      .catch((e) => { if (live) setUnavailable(String((e as { detail?: string })?.detail ?? e)) })
    return () => { live = false }
  }, [project?.id, spec?.id, selectedGraphId, loadRetry, moveTo])

  useEffect(() => {
    setIdent(null)
    setIdentError(null)
    if (!dag || !spec) { setIdentifying(false); return }
    let live = true
    setIdentifying(true)
    api.identify(dag, spec.roles.treatment ?? null, spec.roles.outcome ?? null)
      .then((result) => { if (live) setIdent(result) })
      .catch((e) => { if (live) setIdentError(String((e as { detail?: string })?.detail ?? e)) })
      .finally(() => { if (live) setIdentifying(false) })
    return () => { live = false }
  }, [dag, project?.id, spec?.id, spec?.roles?.treatment, spec?.roles?.outcome, selectedGraphId, identifyRetry])

  if (unavailable) {
    return (
      <div className="canvas-pad">
        <div className="banner caution">
          <span className="grow">This graph could not be opened. {unavailable}</span>
          <button className="btn sm" onClick={() => setLoadRetry((n) => n + 1)}>Retry</button>
        </div>
      </div>
    )
  }
  if (!spec) return <div className="empty-state"><h2>Open a question to draw its graph</h2><p className="hint">Choose or create a question in the project panel.</p></div>
  if (!dag) return <div className="empty-state"><span className="spinner" /> Building a starting graph…</div>

  // Both the node drag and the drop-a-variable handler go through here, so the
  // camera is undone in exactly one place and a drop still lands where the
  // pointer was however far the graph has been zoomed or slid.
  const toSvg = (e: { clientX: number; clientY: number }) => {
    const at = toRoot(svg.current, e.clientX, e.clientY)
    return { x: (at.x - viewport.x) / viewport.k, y: (at.y - viewport.y) / viewport.k }
  }

  const startPan = (e: ReactMouseEvent<SVGSVGElement>) => {
    // Empty canvas only: a mousedown on a node is that node's drag.
    if (e.target !== e.currentTarget) return
    if (glide.current !== null) { cancelAnimationFrame(glide.current); glide.current = null }
    const at = toRoot(svg.current, e.clientX, e.clientY)
    const cur = viewportRef.current
    panFrom.current = { rx: at.x, ry: at.y, x: cur.x, y: cur.y }
    target.current = cur
    setPanning(true)
    // Clicking the canvas should also arm the keyboard shortcuts below.
    svg.current?.focus()
  }

  const endPan = () => {
    if (!panFrom.current) return
    panFrom.current = null
    setPanning(false)
  }

  const panBy = (dx: number, dy: number) => {
    const cur = target.current
    moveTo({ ...cur, x: cur.x + dx, y: cur.y + dy })
  }

  /* Everything the mouse can do to the camera, done from the keyboard. */
  const onKeyDown = (e: ReactKeyboardEvent<SVGSVGElement>) => {
    const nudge = e.shiftKey ? 120 : 45
    switch (e.key) {
      case '+': case '=': zoomAbout(STEP, W / 2, H / 2); break
      case '-': case '_': zoomAbout(1 / STEP, W / 2, H / 2); break
      case '0': moveTo(IDENTITY); break
      case 'f': case 'F': fitTo(dag.nodes); break
      case 'ArrowLeft': panBy(nudge, 0); break
      case 'ArrowRight': panBy(-nudge, 0); break
      case 'ArrowUp': panBy(0, nudge); break
      case 'ArrowDown': panBy(0, -nudge); break
      default: return
    }
    e.preventDefault()
  }

  const addNode = (variable: string, at?: { x: number; y: number }) => {
    if (dag.nodes.some((n) => n.variable === variable)) return
    const id = variable.replace(/[^a-zA-Z0-9_]/g, '_')
    const pos = at ?? { x: 120 + (dag.nodes.length % 5) * 160, y: 100 + Math.floor(dag.nodes.length / 5) * 110 }
    const nodes = [...dag.nodes, { id, variable, label: variable, role: 'other', x: pos.x, y: pos.y }]
    setDag({ ...dag, nodes })
    // A node added from the side panel lands wherever the layout puts it,
    // which on a zoomed-in view can be off the edge of the pane -- and a
    // variable that appears to do nothing is worse than no button. Dropping
    // one with the mouse needs no such rescue: it is already under the cursor.
    if (!at) fitTo(nodes)
  }

  const bad = new Set(((ident?.bad_controls as string[]) ?? []))
  const adjustSets = (ident?.backdoor as { minimal_sets?: string[][] })?.minimal_sets ?? []
  const cycles = (ident?.cycles as string[][]) ?? []

  return (
    <div className="dag-split" ref={split.ref} style={split.style}>
      {split.handle}
      <div className="dag-canvas-wrap col">
        <div className="row dag-toolbar">
          <button className={`btn sm${linkKind === 'directed' ? ' primary' : ''}`}
                  onClick={() => setLinkKind('directed')}>→ causes</button>
          <button className={`btn sm${linkKind === 'bidirected' ? ' primary' : ''}`}
                  onClick={() => setLinkKind('bidirected')}>⇠⇢ unmeasured confounding</button>
          <div className="spacer" />
          <div className="row" style={{ gap: 4 }} role="group" aria-label="Zoom">
            <button className="btn ghost sm" aria-label="Zoom out"
                    title="Zoom out — or the minus key"
                    onClick={() => zoomAbout(1 / STEP, W / 2, H / 2)}>−</button>
            <span className="tiny hint" style={{ minWidth: 38, textAlign: 'center' }}>
              {Math.round(viewport.k * 100)}%
            </span>
            <button className="btn ghost sm" aria-label="Zoom in"
                    title="Zoom in — or the plus key"
                    onClick={() => zoomAbout(STEP, W / 2, H / 2)}>+</button>
            <button className="btn ghost sm"
                    title="Show the whole graph — or the F key"
                    onClick={() => fitTo(dag.nodes)}>Fit</button>
          </div>
          {linkFrom && <span className="chip teal">click a second node to connect</span>}
          <button className="btn ghost sm" onClick={() => { setLinkFrom(null); setSelected(null) }}>Deselect</button>
          <button className="btn primary sm" disabled={savingGraph || !graphDirty} onClick={() => {
            if (!project || savingGraph) return
            const context = graphContext.current
            const snapshot = JSON.stringify(dag)
            setSavingGraph(true)
            void api.saveDag(project.id, dag as unknown as Record<string, unknown>)
              .then(() => {
                if (graphContext.current !== context) return
                setSavedGraph(snapshot)
                toast('success', 'Graph saved with the project.')
                void useStore.getState().refreshProject()
              })
              .catch(() => { if (graphContext.current === context) toast('error', 'Could not save the graph. Your changes are still on this page.') })
              .finally(() => { if (graphContext.current === context) setSavingGraph(false) })
          }}>{savingGraph ? 'Saving graph…' : 'Save graph'}</button>
        </div>

        <p className={`tiny ${graphDirty ? 'warn' : 'hint'}`} role="status" style={{ margin: '0 0 8px' }}>
          {savingGraph ? 'Saving this graph…' : graphDirty
            ? 'Unsaved graph changes. Save the graph before leaving this page.'
            : 'Graph saved with this project.'}
        </p>

        <div
          ref={canvas}
          className="dag-canvas card"
          onDragOver={(e) => { if (e.dataTransfer.types.includes('text/capy-variable')) e.preventDefault() }}
          onDrop={(e) => {
            e.preventDefault()
            const v = e.dataTransfer.getData('text/capy-variable')
            if (v) addNode(v, toSvg(e))
          }}
        >
          <svg
            ref={svg}
            viewBox={`0 0 ${W} ${H}`}
            className="dag-svg"
            tabIndex={0}
            role="application"
            aria-label={
              'Causal graph. Plus and minus zoom, 0 returns to the original size, ' +
              'F shows the whole graph, and the arrow keys move around it.'
            }
            style={{ cursor: panning ? 'grabbing' : 'grab', touchAction: 'none' }}
            onKeyDown={onKeyDown}
            onMouseDown={startPan}
            onMouseMove={(e) => {
              const from = panFrom.current
              if (from) {
                const at = toRoot(svg.current, e.clientX, e.clientY)
                const next = { k: viewport.k, x: from.x + (at.x - from.rx), y: from.y + (at.y - from.ry) }
                target.current = next
                setCamera(next)
                return
              }
              if (!drag) return
              const p = toSvg(e)
              setDag((d) => d && ({
                ...d,
                nodes: d.nodes.map((n) => n.id === drag.id ? { ...n, x: p.x - drag.dx, y: p.y - drag.dy } : n),
              }))
            }}
            onMouseUp={() => { endPan(); setDrag(null) }}
            onMouseLeave={() => { endPan(); setDrag(null) }}
          >
            <defs>
              <marker id="dag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M0 0 L10 5 L0 10 z" fill="var(--muted)" />
              </marker>
            </defs>

            {/* One group carries the camera, so every coordinate below stays in
                the graph's own space and nothing else has to know about zoom. */}
            <g transform={`translate(${viewport.x} ${viewport.y}) scale(${viewport.k})`}>
              {dag.edges.map((e, i) => {
                const a = dag.nodes.find((n) => n.id === e.from)
                const b = dag.nodes.find((n) => n.id === e.to)
                if (!a || !b) return null
                const [x1, y1, x2, y2] = trim(a, b)
                return (
                  <g key={i} className="dag-edge" onClick={() =>
                    setDag((d) => d && ({ ...d, edges: d.edges.filter((_, j) => j !== i) }))}>
                    <line x1={x1} y1={y1} x2={x2} y2={y2}
                          stroke="var(--muted)" strokeWidth={1.8}
                          strokeDasharray={e.kind === 'bidirected' ? '6 4' : undefined}
                          markerEnd={e.kind === 'bidirected' ? undefined : 'url(#dag-arrow)'} />
                    <title>{e.kind === 'bidirected' ? 'Unmeasured common cause' : 'Direct cause'} — click to remove</title>
                  </g>
                )
              })}

              {dag.nodes.map((n) => {
                const isBad = bad.has(n.variable ?? n.id)
                const role = n.role ?? 'other'
                return (
                  <g
                    key={n.id}
                    className={`dag-node role-${role}${selected === n.id ? ' selected' : ''}${isBad ? ' bad' : ''}`}
                    transform={`translate(${n.x},${n.y})`}
                    onMouseDown={(e) => {
                      const p = toSvg(e)
                      setDrag({ id: n.id, dx: p.x - n.x, dy: p.y - n.y })
                    }}
                    onClick={() => {
                      if (linkFrom && linkFrom !== n.id) {
                        setDag((d) => d && ({ ...d, edges: [...d.edges, { from: linkFrom, to: n.id, kind: linkKind }] }))
                        setLinkFrom(null)
                      } else {
                        setLinkFrom(n.id)
                        setSelected(n.id)
                      }
                    }}
                  >
                    <ellipse rx={RX} ry={RY}
                             fill={n.latent ? 'transparent' : 'var(--paper-raised)'}
                             stroke={isBad ? 'var(--crimson)' : roleColor(role)}
                             strokeWidth={selected === n.id ? 2.6 : 1.8}
                             strokeDasharray={n.latent ? '5 4' : undefined} />
                    <text textAnchor="middle" dy="4" fontSize="13" fill="var(--ink)">
                      {(n.label ?? n.variable ?? n.id).slice(0, 12)}
                    </text>
                    <title>{role}{isBad ? ' — descendant of treatment: do not adjust for this' : ''}</title>
                  </g>
                )
              })}
            </g>

          </svg>
          <p className="dag-caption">{dag.caption ?? 'This graph is an assumption you are making.'}</p>
        </div>

        <p className="tiny hint">
          Drag a variable from the data sheet onto the canvas. Click one node then another to connect them;
          click an edge to remove it. Drag nodes to rearrange.
          Scroll to zoom and drag the empty background to move around — or click the canvas once and use
          the arrow keys, plus and minus, F to show the whole graph and 0 to go back to the original size.
        </p>
      </div>

      <aside className="dag-side scroll">
        <section className="card pad">
          <div className="panel-title">Identification</div>
          {identifying && <p className="hint" role="status">Checking what this graph can identify…</p>}
          {identError && <div className="banner caution" role="alert">
            <span className="grow">The graph could not be checked. {identError}</span>
            <button className="btn sm" onClick={() => setIdentifyRetry((n) => n + 1)}>Retry</button>
          </div>}
          {ident && <>
          {cycles.length > 0 && (
            <div className="banner error">
              <span>
                This graph has a cycle ({cycles[0].join(' → ')}). A causal graph has to be acyclic; nothing
                can be identified until you remove one of those arrows.
              </span>
            </div>
          )}
          {!cycles.length && (
            <>
              <p className="tiny hint">
                Back-door adjustment sets for {spec?.roles?.treatment ?? 'treatment'} →{' '}
                {spec?.roles?.outcome ?? 'outcome'}:
              </p>
              {adjustSets.length === 0 && (
                <div className="banner caution">
                  <span>
                    {(ident?.backdoor as { reason?: string })?.reason ??
                      'No sufficient adjustment set exists in this graph. The effect is not identified by adjustment alone.'}
                  </span>
                </div>
              )}
              <ul className="adjust-sets">
                {adjustSets.map((set, i) => (
                  <li key={i}>
                    <span className="grow">{set.length ? set.join(', ') : '(the empty set)'}</span>
                    <button className="btn sm" onClick={() => {
                      setRole('confounders', set, 'Adopted an adjustment set from the graph')
                      toast('info', 'Confounder list updated from the graph. You can still edit it by hand.')
                    }}>
                      Use this set
                    </button>
                  </li>
                ))}
              </ul>
              {((ident?.frontdoor as { identified?: boolean })?.identified) && (
                <div className="banner info">
                  <span>A front-door path is available in this graph as well.</span>
                </div>
              )}
              {((ident?.instruments as string[]) ?? []).length > 0 && (
                <p className="tiny">
                  Candidate instruments: {(ident!.instruments as string[]).join(', ')}
                </p>
              )}
            </>
          )}
          </>}
        </section>

        {bad.size > 0 && (
          <section className="card pad">
            <div className="panel-title">Do not adjust for these</div>
            <p className="tiny hint">
              Descendants of the treatment. Adjusting for one blocks part of the very effect you are
              measuring.
            </p>
            <div className="row wrap" style={{ gap: 4 }}>
              {[...bad].map((b) => <span key={b} className="chip crimson">{b}</span>)}
            </div>
          </section>
        )}

        <section className="card pad">
          <div className="panel-title">Add a variable</div>
          <select aria-label="Add a variable to the graph" defaultValue="" onChange={(e) => { if (e.target.value) { addNode(e.target.value); e.target.value = '' } }}>
            <option value="">Choose…</option>
            {columns.filter((c) => !dag.nodes.some((n) => n.variable === c.name))
              .map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
          </select>
          <button className="btn sm" style={{ marginTop: 8 }} onClick={() => {
            const nodes = [...dag.nodes, {
              id: `U${dag.nodes.length}`, variable: null, label: 'U', role: 'latent', latent: true,
              x: 500, y: 480,
            }]
            setDag({ ...dag, nodes })
            fitTo(nodes)
          }}>
            Add an unmeasured cause
          </button>
        </section>

        <section className="card pad">
          <div className="panel-title">Structure learning</div>
          <p className="tiny hint">
            Automatic structure discovery is not available in this release. Draw the assumptions you can
            defend, then use the identification checks above to inspect their implications.
          </p>
        </section>
      </aside>
    </div>
  )
}

function normalise(d: Dag): Dag {
  const nodes = (d.nodes ?? []).map((n, i) => ({
    ...n,
    x: n.x ?? 140 + (i % 5) * 170,
    y: n.y ?? 110 + Math.floor(i / 5) * 120,
  }))
  return { ...d, nodes, edges: d.edges ?? [] }
}

function trim(a: Node, b: Node): [number, number, number, number] {
  const dx = b.x - a.x
  const dy = b.y - a.y
  const len = Math.hypot(dx, dy) || 1
  const ux = dx / len
  const uy = dy / len
  const scale = (rx: number, ry: number) => rx * ry / Math.hypot(ry * ux, rx * uy)
  const ra = scale(RX + 2, RY + 2)
  const rb = scale(RX + 8, RY + 8)
  return [a.x + ux * ra, a.y + uy * ra, b.x - ux * rb, b.y - uy * rb]
}

function roleColor(role: string): string {
  return ({
    treatment: 'var(--teal)', outcome: 'var(--clay)', confounder: 'var(--dusk)',
    mediator: 'var(--plum)', instrument: 'var(--moss)', collider: 'var(--ochre)',
    latent: 'var(--muted)',
  } as Record<string, string>)[role] ?? 'var(--rule-strong)'
}
