/** A pop-up that stays inside the window.
 *
 * A menu positioned with `absolute; top: 100%` sits wherever its button
 * happens to be, and the run picker's button is most of the way down the
 * screen: its menu ran 114px past the bottom edge on a 900px window and 274px
 * on a 720px one, with no internal scroll, so the runs at the end of the list
 * could not be reached at all.
 *
 * This measures the button, opens the menu upwards when there is more room
 * that way, and caps its height to the space actually available so the
 * overflow becomes a scrollbar rather than a clipped edge. It also closes on
 * an outside click and on Escape, which the run picker never did.
 */

import {
  useCallback, useEffect, useLayoutEffect, useRef, useState,
  type CSSProperties, type RefObject,
} from 'react'

export interface AnchoredMenu {
  /** Put this on the button that opens the menu. */
  anchorRef: RefObject<HTMLButtonElement | null>
  /** Put this on the menu itself, along with `style`. */
  menuRef: RefObject<HTMLDivElement | null>
  style: CSSProperties
}

export function useAnchoredMenu(
  open: boolean,
  onClose: () => void,
  options?: { width?: number; minHeight?: number; maxHeight?: number },
): AnchoredMenu {
  const anchorRef = useRef<HTMLButtonElement | null>(null)
  const menuRef = useRef<HTMLDivElement | null>(null)
  // Hidden until it has been placed, so it never paints in the wrong spot first.
  const [style, setStyle] = useState<CSSProperties>({ visibility: 'hidden' })

  const width = options?.width
  const minHeight = options?.minHeight ?? 140
  const maxHeight = options?.maxHeight ?? 360

  useLayoutEffect(() => {
    if (!open) { setStyle({ visibility: 'hidden' }); return }
    const place = () => {
      const anchor = anchorRef.current
      if (!anchor) return
      const r = anchor.getBoundingClientRect()
      const gap = 4
      const margin = 10
      const below = window.innerHeight - r.bottom - gap - margin
      const above = r.top - gap - margin
      // Only flip when there is genuinely too little room below and more above.
      const flip = below < minHeight && above > below
      const room = Math.max(minHeight, flip ? above : below)
      setStyle({
        position: 'fixed',
        right: Math.max(margin, window.innerWidth - r.right),
        ...(flip
          ? { bottom: window.innerHeight - r.top + gap }
          : { top: r.bottom + gap }),
        maxHeight: Math.min(room, maxHeight),
        ...(width ? { width } : null),
      })
    }
    place()
    // Anything that moves the button moves the menu: a resize, or a scroll in
    // any of the panes it might be sitting in.
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    return () => {
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
    }
  }, [open, width, minHeight, maxHeight])

  const close = useCallback(onClose, [onClose])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node
      if (menuRef.current?.contains(target) || anchorRef.current?.contains(target)) return
      close()
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open, close])

  return { anchorRef, menuRef, style }
}
