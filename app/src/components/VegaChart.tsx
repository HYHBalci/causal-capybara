/** Every plot in the product renders through here.
 *
 * Adapters emit a Vega-Lite spec in the light palette; this component merges a
 * dark config at render time so one stored spec survives both themes and the
 * export. Hover gives exact values; the caption and the "what would worry me"
 * line travel with the plot, and the underlying data is downloadable, because a
 * diagnostic you cannot interrogate is decoration.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import embed, { type Result as EmbedResult, type VisualizationSpec } from 'vega-embed'
import { expressionInterpreter } from 'vega-interpreter'
import type { Artifact } from '../types'
import { count, measure } from '../format'
import { saveFile } from '../desktop'

const DARK = {
  background: 'transparent',
  axis: {
    labelColor: '#a49c92',
    titleColor: '#f2ede6',
    gridColor: '#3a3634',
    domainColor: '#3a3634',
    tickColor: '#3a3634',
  },
  legend: { labelColor: '#a49c92', titleColor: '#f2ede6' },
  title: { color: '#f2ede6' },
  view: { stroke: 'transparent' },
}

/** The daylight palette, and what each colour becomes at night.
 *
 * Merging a dark `config` re-colours axes, legends and titles, because those
 * are config-level settings. It cannot reach a colour written on a mark, in a
 * scale range, or inside a conditional encoding -- and those are where a
 * plot's actual ink lives. So a forest plot's points kept a paper-white halo
 * on a dark ground, and the missingness heatmap ramped from a light cream that
 * was no longer the background.
 *
 * Rewriting the literals is the only thing that reaches all three, and doing it
 * here means an adapter or a screen can go on emitting one spec in the light
 * palette, which is also the palette an exported PNG should be drawn in.
 * The keys are the tokens from theme.css; the values are their night halves.
 */
const NIGHT: Record<string, string> = {
  '#faf7f2': '#1a1816',   // paper
  '#ffffff': '#232120',   // paper-raised
  '#f2ede4': '#131211',   // paper-sunken
  '#1c1917': '#f2ede6',   // ink
  '#453f39': '#d9d2c9',   // ink-soft
  '#5f5850': '#a49c92',   // muted
  '#6f6862': '#a49c92',   // muted, as it was before the palette was retuned
  '#726b63': '#9a9289',   // faint
  '#948c83': '#9a9289',   // faint, previous value
  '#ded7cc': '#3a3634',   // rule
  '#8e8578': '#7d766e',   // rule-strong
  '#c9c0b2': '#7d766e',   // rule-strong, previous value
  '#0e7c86': '#3fb0ba',   // teal
  '#0a5e66': '#6bcbd3',   // teal-strong
  '#e2f1f2': '#123033',   // teal-soft
  '#c67a16': '#e0a34c',   // ochre
  '#fbf0dc': '#322614',   // ochre-soft
  '#b3261e': '#e0665e',   // crimson
  '#3f7a52': '#6aa87c',   // moss
  '#4c6e9c': '#7d9dc9',   // dusk
  '#7a6b9b': '#a396c2',   // plum
  '#a8553a': '#c97d62',   // clay
  '#8c8378': '#a49c92',   // stone
}

/** Rewrite every palette colour in a spec, wherever it appears. */
function toNight(value: unknown): unknown {
  if (typeof value === 'string') {
    const night = NIGHT[value.toLowerCase()]
    return night ?? value
  }
  if (Array.isArray(value)) return value.map(toNight)
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) out[k] = toNight(v)
    return out
  }
  return value
}

function isDark(): boolean {
  const attr = document.documentElement.getAttribute('data-theme')
  if (attr === 'dark') return true
  if (attr === 'light') return false
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
}

function deepMerge<T extends Record<string, unknown>>(base: T, patch: Record<string, unknown>): T {
  const out: Record<string, unknown> = { ...base }
  for (const [k, v] of Object.entries(patch)) {
    const cur = out[k]
    out[k] = v && typeof v === 'object' && !Array.isArray(v) && cur && typeof cur === 'object' && !Array.isArray(cur)
      ? deepMerge(cur as Record<string, unknown>, v as Record<string, unknown>)
      : v
  }
  return out as T
}

export function VegaChart({
  spec,
  height,
  onError,
}: {
  spec: Record<string, unknown>
  height?: number
  onError?: (message: string) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const view = useRef<EmbedResult | null>(null)
  const [ready, setReady] = useState(false)
  const [saving, setSaving] = useState(false)
  const [exportStatus, setExportStatus] = useState<string | null>(null)
  const [exportError, setExportError] = useState<string | null>(null)
  const [dark, setDark] = useState(isDark)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)')
    const update = () => setDark(isDark())
    mq?.addEventListener('change', update)
    const obs = new MutationObserver(update)
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => {
      mq?.removeEventListener('change', update)
      obs.disconnect()
    }
  }, [])

  const themed = useMemo(() => {
    let base = JSON.parse(JSON.stringify(spec)) as Record<string, unknown>
    if (dark) {
      // Remap the ink first, then merge the axis and legend settings on top --
      // the remap must not be applied to DARK's own colours, which are already
      // night values and would otherwise be looked up and left alone by luck
      // rather than by design.
      base = toNight(base) as Record<string, unknown>
      base.config = deepMerge((base.config as Record<string, unknown>) ?? {}, DARK)
    }
    if (height) base.height = height
    return base as VisualizationSpec
  }, [spec, dark, height])

  useEffect(() => {
    let cancelled = false
    if (!host.current) return
    setError(null)
    setReady(false)
    setExportStatus(null)
    setExportError(null)
    embed(host.current, themed, {
      actions: false,
      renderer: 'canvas',
      downloadFileName: 'causal-capybara-plot',
      // Vega normally compiles spec expressions into JavaScript with `Function`,
      // which the packaged app's Content Security Policy refuses -- every plot
      // failed with "unsafe-eval is not an allowed source of script". Parsing to
      // an AST and walking it with Vega's own interpreter is the supported way
      // out, and keeps the policy strict instead of opening it up.
      ast: true,
      expr: expressionInterpreter,
    })
      .then((res) => {
        if (cancelled) {
          res.finalize()
          return
        }
        view.current?.finalize()
        view.current = res
        setReady(true)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        onError?.(message)
      })
    return () => {
      cancelled = true
      view.current?.finalize()
      view.current = null
    }
  }, [themed])

  const exportPlot = async (format: 'png' | 'svg') => {
    const chart = view.current
    if (!chart || saving) return
    setSaving(true)
    setExportStatus(null)
    setExportError(null)
    try {
      // Serialize directly: fetching a data URL is blocked by the desktop CSP.
      const blob = format === 'svg'
        ? new Blob([await chart.view.toSVG()], { type: 'image/svg+xml' })
        : await new Promise<Blob>((resolve, reject) => {
          chart.view.toCanvas(2).then((canvas) => {
            canvas.toBlob((image) => image ? resolve(image) : reject(new Error('PNG encoding failed.')), 'image/png')
          }).catch(reject)
        })
      const path = await saveFile(blob, `causal-capybara-plot.${format}`, [
        { name: format.toUpperCase() + ' image', extensions: [format] },
      ])
      if (path) setExportStatus(`Saved ${path}`)
    } catch (err) {
      setExportError(`Could not save the plot: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setSaving(false)
    }
  }

  if (error) {
    return (
      <div className="banner error" role="alert">
        <span>This plot could not be drawn: {error}</span>
      </div>
    )
  }
  return (
    <div className="chart">
      <div ref={host} className="vega-host" style={{ width: '100%' }} />
      <div className="row wrap" style={{ marginTop: 4 }}>
        <button className="btn ghost sm" disabled={!ready || saving} onClick={() => void exportPlot('png')}>
          Save PNG
        </button>
        <button className="btn ghost sm" disabled={!ready || saving} onClick={() => void exportPlot('svg')}>
          Save SVG
        </button>
        {saving && <span className="tiny hint" role="status">Saving…</span>}
      </div>
      {exportStatus && <p className="tiny hint" role="status">{exportStatus}</p>}
      {exportError && <div className="banner error" role="alert">{exportError}</div>}
    </div>
  )
}

/** An artifact with its caption, its Explain link, and a way to get the data out. */
export function ArtifactView({
  artifact,
  height,
  onExplain,
  compact,
}: {
  artifact: Artifact
  height?: number
  onExplain?: (key: string) => void
  compact?: boolean
}) {
  if (artifact.kind === 'vega' && artifact.spec) {
    return (
      <figure className="artifact">
        {!compact && artifact.title && <figcaption className="artifact-title">{artifact.title}</figcaption>}
        <VegaChart spec={artifact.spec} height={height} />
        {artifact.caption && <p className="artifact-caption">{artifact.caption}</p>}
        {artifact.explain_key && onExplain && (
          <button className="btn ghost sm explain-chip" onClick={() => onExplain(artifact.explain_key!)}>
            Explain this
          </button>
        )}
      </figure>
    )
  }
  if (artifact.kind === 'table' && Array.isArray(artifact.data)) {
    return <TableArtifact artifact={artifact} />
  }
  if (artifact.kind === 'text') {
    return (
      <pre className="mono classic-block" tabIndex={0} aria-label={artifact.title ?? "Result text"}>{String(artifact.data ?? '')}</pre>
    )
  }
  return null
}

function TableArtifact({ artifact }: { artifact: Artifact }) {
  const rows = (artifact.data as Record<string, unknown>[]) ?? []
  const cols = artifact.columns ?? Array.from(new Set(rows.flatMap((r) => Object.keys(r))))
  if (!rows.length) return null
  return (
    <figure className="artifact">
      {artifact.title && <figcaption className="artifact-title">{artifact.title}</figcaption>}
      <div className="scroll" style={{ maxHeight: 320 }} tabIndex={0} role="region" aria-label={artifact.title ?? "Result table"}>
        <table className="grid">
          <thead>
            <tr>{cols.map((c) => <th key={c}>{humanise(c)}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {cols.map((c) => (
                  <td key={c} className={typeof r[c] === 'number' ? 'n' : undefined}>
                    {fmtCell(r[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: 4 }}>
        <button className="btn ghost sm" onClick={() => downloadCsv(artifact.title ?? 'table', cols, rows)}>
          Download CSV
        </button>
      </div>
      {artifact.caption && <p className="artifact-caption">{artifact.caption}</p>}
    </figure>
  )
}

export function humanise(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\bsmd\b/gi, 'SMD')
    .replace(/\bess\b/gi, 'ESS')
    .replace(/\bci\b/gi, 'CI')
    .replace(/\bse\b/gi, 'SE')
    .replace(/\bps\b/gi, 'PS')
    .replace(/\bp value\b/gi, 'p')
    .replace(/^./, (c) => c.toUpperCase())
}

export function fmtCell(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return fmtNum(v)
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  return String(v)
}

/** Every estimate, standard error and bound in the interface comes through
 *  here. It used to call toLocaleString() with no locale, so on a Dutch machine
 *  1542.8 rendered as "1.543" beside a count of "2,675" -- a thousands
 *  separator that reads as a decimal point, in a table of effect sizes.
 *  format.ts owns the one locale the app uses; this just defers to it. */
export function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return count(v)
  return measure(v, digits)
}

function downloadCsv(name: string, cols: string[], rows: Record<string, unknown>[]) {
  const esc = (v: unknown) => {
    const s = v === null || v === undefined ? '' : String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${name.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}.csv`
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}
