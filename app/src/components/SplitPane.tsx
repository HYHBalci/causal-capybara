/** A draggable divider for the two-column canvases.
 *
 * The board, the diagnostics, the graph editor, the simulation lab and the
 * report are each one grid of two tracks whose ratio was fixed in the
 * stylesheet. This turns the first track into a remembered width and puts a
 * divider on the boundary; the second track takes whatever is left.
 *
 * It adds nothing to the tree but the handle -- no wrapper around the panes --
 * so the grid keeps working exactly as it did, and below the width where the
 * stylesheet stacks the two into one column the handle stays away.
 *
 * The handle is placed where the boundary actually is, measured, rather than
 * where the default says it should be. Those are different numbers whenever
 * the panes around the canvas have themselves been resized: placing the graph
 * editor's divider at its nominal 860px once put it beyond the right-hand edge
 * of the canvas entirely, on top of the inspector, where it could not be
 * grabbed at all.
 */

import {
  useCallback, useEffect, useLayoutEffect, useState,
  type CSSProperties, type ReactNode,
} from 'react'
import { Resizer, usePanelSize } from './Resizer'

export function useSplitPane(
  storageKey: string,
  defaultLeft: number,
  minLeft = 240,
): {
  ref: (node: HTMLDivElement | null) => void
  style: CSSProperties | undefined
  handle: ReactNode
} {
  // A callback ref, not a plain one: the graph editor and the report mount
  // their split only once there is something to show, and an effect keyed on a
  // ref object runs before that and never again.
  const [node, setNode] = useState<HTMLDivElement | null>(null)
  const ref = useCallback((el: HTMLDivElement | null) => setNode(el), [])
  const maxLeft = () => Math.max(minLeft + 120, Math.round(window.innerWidth * 0.72))
  const { size, setSize, reset } = usePanelSize(storageKey, null, minLeft, maxLeft)
  const [stacked, setStacked] = useState(
    () => typeof window !== 'undefined' && window.innerWidth <= 1100,
  )
  /** The first track's real width right now. */
  const [boundary, setBoundary] = useState<number | null>(null)

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 1100px)')
    const on = () => setStacked(mq.matches)
    on()
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])

  useLayoutEffect(() => {
    const el = node
    if (!el || stacked) { setBoundary(null); return }
    const first = Array.from(el.children).find(
      (c) => !c.classList.contains('resizer'),
    ) as HTMLElement | undefined
    if (!first) return
    const measure = () => setBoundary(first.offsetWidth)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(first)
    ro.observe(el)
    return () => ro.disconnect()
  }, [node, stacked, size])

  if (stacked) return { ref, style: undefined, handle: null }

  const at = boundary ?? size ?? defaultLeft
  return {
    ref,
    style: size ? { gridTemplateColumns: `${size}px minmax(0, 1fr)` } : undefined,
    handle: (
      <Resizer
        axis="x"
        side="left"
        label="Panel width"
        current={at}
        onChange={setSize}
        onReset={reset}
        min={minLeft}
        max={maxLeft}
        style={{ left: at - 3, right: 'auto', top: 0, bottom: 0, width: 7 }}
      />
    ),
  }
}
