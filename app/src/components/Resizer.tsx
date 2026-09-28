/** Draggable dividers, so the panes are yours to size.
 *
 * Every pane in the studio had a width or a height chosen for it: the
 * navigator 232px, the inspector 300px, the variable rail 236px, the data
 * health panel 42% of the sheet, the job bar however tall its list happened to
 * be. On a wide screen that wastes space, on a narrow one it starves the thing
 * you are actually reading, and a forest plot with eight methods in it needs
 * more room than a table of two.
 *
 * A size, once dragged, is remembered per pane and survives a restart. Double
 * click a divider to put it back. The dividers are real separators: they take
 * focus, arrow keys nudge them, Home and End go to the extremes, so none of
 * this depends on being able to hit a four-pixel target with a mouse.
 */

import {
  useCallback, useEffect, useRef, useState,
  type CSSProperties, type PointerEvent as ReactPointerEvent, type KeyboardEvent as ReactKeyboardEvent,
} from 'react'

const PREFIX = 'capy.panel.'

type Bound = number | (() => number)
const value = (b: Bound): number => (typeof b === 'function' ? b() : b)

/** A remembered pane size in pixels. `null` means "however big the content is",
 *  which is the right default for the job bar and stays that way until dragged. */
export function usePanelSize(
  key: string,
  fallback: number | null,
  min: Bound,
  max: Bound,
): {
  size: number | null
  setSize: (n: number | null) => void
  reset: () => void
  isCustom: boolean
} {
  const clamp = useCallback(
    (n: number) => Math.round(Math.min(value(max), Math.max(value(min), n))),
    [min, max],
  )

  const [size, setRaw] = useState<number | null>(() => {
    try {
      const raw = localStorage.getItem(PREFIX + key)
      if (raw === null) return fallback
      const n = Number(raw)
      return Number.isFinite(n) ? clamp(n) : fallback
    } catch {
      // Private windows and locked-down profiles throw on access, and a pane
      // that cannot remember its size should still work.
      return fallback
    }
  })
  const [isCustom, setCustom] = useState<boolean>(() => {
    try { return localStorage.getItem(PREFIX + key) !== null } catch { return false }
  })

  const setSize = useCallback((n: number | null) => {
    if (n === null) {
      setRaw(fallback); setCustom(false)
      try { localStorage.removeItem(PREFIX + key) } catch { /* nothing to do */ }
      return
    }
    const next = clamp(n)
    setRaw(next); setCustom(true)
    try { localStorage.setItem(PREFIX + key, String(next)) } catch { /* nothing to do */ }
  }, [key, fallback, clamp])

  const reset = useCallback(() => setSize(null), [setSize])
  return { size, setSize, reset, isCustom }
}

/** The element's size right now, kept current as it changes.
 *
 * A pane with no explicit size still has a real one, and that -- not the
 * default it would have taken -- is where a drag has to start from.
 */
export function useMeasured<T extends HTMLElement>(
  axis: 'x' | 'y',
): [(node: T | null) => void, number | null] {
  const [node, setNode] = useState<T | null>(null)
  const [size, setSize] = useState<number | null>(null)
  const ref = useCallback((n: T | null) => setNode(n), [])
  useEffect(() => {
    if (!node) { setSize(null); return }
    const measure = () => setSize(axis === 'x' ? node.offsetWidth : node.offsetHeight)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(node)
    return () => ro.disconnect()
  }, [node, axis])
  return [ref, size]
}

interface ResizerProps {
  /** 'x' drags left and right, 'y' up and down. */
  axis: 'x' | 'y'
  /** The size the pane has right now, in pixels. */
  current: number
  onChange: (next: number) => void
  onReset?: () => void
  /** Dragging away from the pane makes it bigger by default. Set this when the
   *  pane sits on the right or the bottom, where the sense is reversed. */
  invert?: boolean
  min: Bound
  max: Bound
  label: string
  /** Where the handle sits inside its (positioned) parent. */
  side: 'left' | 'right' | 'top' | 'bottom'
  step?: number
  /** Overrides the placement, for handles that sit on a track boundary rather
   *  than at the edge of their container. */
  style?: CSSProperties
}

export function Resizer({
  axis, current, onChange, onReset, invert, min, max, label, side, step = 16,
  style: styleOverride,
}: ResizerProps) {
  const [dragging, setDragging] = useState(false)
  const start = useRef({ pointer: 0, size: 0 })

  const clamp = useCallback(
    (n: number) => Math.round(Math.min(value(max), Math.max(value(min), n))),
    [min, max],
  )

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    e.preventDefault()
    // Capture, so the drag survives crossing a canvas, an iframe or the window
    // edge -- without it the divider is dropped the moment the pointer leaves.
    e.currentTarget.setPointerCapture(e.pointerId)
    start.current = { pointer: axis === 'x' ? e.clientX : e.clientY, size: current }
    setDragging(true)
  }

  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragging) return
    const delta = (axis === 'x' ? e.clientX : e.clientY) - start.current.pointer
    onChange(clamp(start.current.size + (invert ? -delta : delta)))
  }

  const stop = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragging) return
    try { e.currentTarget.releasePointerCapture(e.pointerId) } catch { /* already gone */ }
    setDragging(false)
  }

  // While dragging, stop the pointer selecting text across the whole window and
  // keep the resize cursor even where it passes over other elements.
  useEffect(() => {
    if (!dragging) return
    const body = document.body
    const prevSelect = body.style.userSelect
    const prevCursor = body.style.cursor
    body.style.userSelect = 'none'
    body.style.cursor = axis === 'x' ? 'col-resize' : 'row-resize'
    return () => { body.style.userSelect = prevSelect; body.style.cursor = prevCursor }
  }, [dragging, axis])

  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const grow = invert ? -1 : 1
    const keys: Record<string, number | 'min' | 'max'> = {
      ArrowLeft: axis === 'x' ? -step * grow : 0,
      ArrowRight: axis === 'x' ? step * grow : 0,
      ArrowUp: axis === 'y' ? -step * grow : 0,
      ArrowDown: axis === 'y' ? step * grow : 0,
      Home: 'min',
      End: 'max',
    }
    const move = keys[e.key]
    if (move === undefined || move === 0) return
    e.preventDefault()
    if (move === 'min') onChange(value(min))
    else if (move === 'max') onChange(value(max))
    else onChange(clamp(current + move))
  }

  const base: CSSProperties =
    side === 'left' ? { left: -3, top: 0, bottom: 0, width: 7, cursor: 'col-resize' }
    : side === 'right' ? { right: -3, top: 0, bottom: 0, width: 7, cursor: 'col-resize' }
    : side === 'top' ? { top: -3, left: 0, right: 0, height: 7, cursor: 'row-resize' }
    : { bottom: -3, left: 0, right: 0, height: 7, cursor: 'row-resize' }
  const style: CSSProperties = { ...base, ...styleOverride }

  return (
    <div
      className={`resizer resizer-${axis}${dragging ? ' dragging' : ''}`}
      style={style}
      role="separator"
      tabIndex={0}
      aria-label={label}
      aria-orientation={axis === 'x' ? 'vertical' : 'horizontal'}
      aria-valuenow={Math.round(current)}
      aria-valuemin={Math.round(value(min))}
      aria-valuemax={Math.round(value(max))}
      title={`Drag to resize ${label.toLowerCase()}. Double click to reset.`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={stop}
      onPointerCancel={stop}
      onKeyDown={onKeyDown}
      onDoubleClick={() => onReset?.()}
    />
  )
}
