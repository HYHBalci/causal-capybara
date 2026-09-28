/** One place that decides how a number looks.
 *
 * `toLocaleString()` with no locale follows whatever the machine is set to, so
 * the same build showed "1.020" in the status bar (Dutch grouping) next to
 * "2,008" on a chart axis (Vega's own en-US default) -- in the same window, for
 * numbers of the same kind. Charts come from the engine and are not going to
 * negotiate, so the interface matches them rather than the operating system.
 *
 * A study is read by people in more than one country and the numbers in it have
 * to mean the same thing to all of them. That is worth more here than matching
 * the reader's regional settings.
 */

const LOCALE = 'en-GB'

/** A count: rows, units, periods. Grouped, never decimalised. */
export function count(n: number | null | undefined, fallback = '—'): string {
  if (n == null || !Number.isFinite(n)) return fallback
  return Math.round(n).toLocaleString(LOCALE)
}

/** A year, a period index, an id: whole, and never grouped. 2008, not 2,008. */
export function whole(n: number | null | undefined, fallback = '—'): string {
  if (n == null || !Number.isFinite(n)) return fallback
  return String(Math.round(n))
}

/** An estimate, a standard error, a bound. Significant digits, not decimals. */
export function measure(n: number | null | undefined, digits = 4, fallback = '—'): string {
  if (n == null || !Number.isFinite(n)) return fallback
  if (n === 0) return '0'
  const magnitude = Math.abs(n)
  if (magnitude >= 1e6 || magnitude < 1e-4) return n.toExponential(2)
  return Number(n.toPrecision(digits)).toLocaleString(LOCALE, { maximumSignificantDigits: digits })
}

/** A share in [0, 1] shown as a percentage. */
export function percent(n: number | null | undefined, digits = 1, fallback = '—'): string {
  if (n == null || !Number.isFinite(n)) return fallback
  return `${(n * 100).toLocaleString(LOCALE, { maximumFractionDigits: digits })}%`
}
