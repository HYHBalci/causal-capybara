import type { ColumnMeta } from './types'

/** Match design roles using inferred kinds and genuine numeric storage.
 * Named categories and string identifiers remain usable in matching roles. */
export function canAssignColumn(column: ColumnMeta, accepts?: readonly string[]): boolean {
  if (!accepts?.length || accepts.includes(column.kind)) return true
  // Kind is a heuristic: a distinct integer duration looks like an ID, while
  // a discrete running score looks categorical. Both are still numeric inputs.
  // Boolean, datetime and string storage must not acquire numeric eligibility.
  return accepts.includes('continuous')
    && /^(?:u?int(?:8|16|32|64)|float(?:16|32|64|128)|double)(?:\[pyarrow\])?$/i.test(column.dtype)
}

export function roleColumnSelection(raw: unknown): string[] {
  return (Array.isArray(raw) ? raw : [raw]).filter((value): value is string => typeof value === 'string' && !!value)
}

/** Existing assignments stay visible so changed kinds/designs never prevent removal. */
export function roleColumnOptions(columns: ColumnMeta[], accepts: readonly string[] | undefined, raw: unknown): ColumnMeta[] {
  const selected = new Set(roleColumnSelection(raw))
  return columns.filter((column) => selected.has(column.name) || canAssignColumn(column, accepts))
}
