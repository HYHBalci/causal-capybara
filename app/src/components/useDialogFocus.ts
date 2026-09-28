import { useEffect, type RefObject } from 'react'

export function useDialogFocus(ref: RefObject<HTMLElement | null>, open: boolean, close: () => void) {
  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    const element = ref.current
    if (!element) return
    const focusable = () => [...element.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], input, select, textarea, [tabindex="0"]')]
      .filter((item) => item.getClientRects().length > 0)
    focusable()[0]?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close() }
      if (event.key !== 'Tab') return
      const items = focusable()
      const first = items[0], last = items[items.length - 1]
      if (!first) { event.preventDefault(); return }
      if (event.shiftKey && (document.activeElement === first || !element.contains(document.activeElement))) {
        event.preventDefault(); last.focus()
      } else if (!event.shiftKey && (document.activeElement === last || !element.contains(document.activeElement))) {
        event.preventDefault(); first.focus()
      }
    }
    element.addEventListener('keydown', onKey)
    return () => { element.removeEventListener('keydown', onKey); previous?.isConnected && previous.focus() }
  }, [open, ref, close])
}
