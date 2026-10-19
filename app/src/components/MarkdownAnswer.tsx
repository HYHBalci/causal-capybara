import type { ReactNode } from 'react'
import { ExternalLink } from './References'

/** A small text-only Markdown view. HTML and images never execute or fetch. */
function inline(text: string): ReactNode[] {
  const pieces: ReactNode[] = []
  const expression = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\((?:<https?:\/\/[^\s<>]+>|https?:\/\/[^\s)]+)\))/g
  let start = 0
  for (const match of text.matchAll(expression)) {
    const index = match.index!
    pieces.push(text.slice(start, index))
    const token = match[0]
    if (token.startsWith('`')) pieces.push(<code key={index}>{token.slice(1, -1)}</code>)
    else if (token.startsWith('**')) pieces.push(<strong key={index}>{token.slice(2, -2)}</strong>)
    else {
      const link = /^\[([^\]]+)\]\((.+)\)$/.exec(token)!
      const href = link[2].startsWith('<') ? link[2].slice(1, -1) : link[2]
      pieces.push(<ExternalLink key={index} href={href}>{link[1]}</ExternalLink>)
    }
    start = index + token.length
  }
  pieces.push(text.slice(start))
  return pieces
}

export function MarkdownAnswer({ markdown }: { markdown: string }) {
  const lines = markdown.replace(/\r\n?/g, '\n').split('\n')
  const blocks: ReactNode[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    const key = i
    if (!line.trim()) { i++; continue }
    if (/^\s*```/.test(line)) {
      const code: string[] = []; i++
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++])
      i++
      blocks.push(<pre key={key} tabIndex={0} aria-label="Code excerpt"><code>{code.join('\n')}</code></pre>)
      continue
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(line)
    if (heading) {
      const Tag = `h${Math.min(6, heading[1].length + 2)}` as 'h3'
      blocks.push(<Tag key={key}>{inline(heading[2])}</Tag>); i++; continue
    }
    if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) { blocks.push(<hr key={key} />); i++; continue }
    if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[i + 1])) {
      const cells = (row: string) => row.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim())
      const headers = cells(line); const rows: string[][] = []; i += 2
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) rows.push(cells(lines[i++]))
      blocks.push(<div className="markdown-table" key={key} tabIndex={0} role="region" aria-label="Answer table"><table><thead><tr>{headers.map((cell, col) => <th key={col}>{inline(cell)}</th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{headers.map((_, col) => <td key={col}>{inline(row[col] ?? '')}</td>)}</tr>)}</tbody></table></div>)
      continue
    }
    const bullet = /^\s*(?:[-*+]\s+|\d+[.)]\s+)/.exec(line)
    if (bullet) {
      const ordered = /^\s*\d/.test(line)
      const pattern = ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*+]\s+/
      const items: ReactNode[] = []
      while (i < lines.length && pattern.test(lines[i])) { items.push(<li key={i}>{inline(lines[i].replace(pattern, ''))}</li>); i++ }
      blocks.push(ordered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>); continue
    }
    if (/^>\s?/.test(line)) {
      const quote: string[] = []
      while (i < lines.length && /^>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, ''))
      blocks.push(<blockquote key={key}>{inline(quote.join('\n'))}</blockquote>); continue
    }
    const paragraph = [line]; i++
    while (i < lines.length && lines[i].trim() && !/^(?:#{1,6}\s|\s*```|\s*[-*+]\s|\s*\d+[.)]\s|>)/.test(lines[i]) && !(lines[i].includes('|') && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1] ?? ''))) paragraph.push(lines[i++])
    blocks.push(<p key={key}>{inline(paragraph.join('\n'))}</p>)
  }
  return <div className="markdown-answer">{blocks}</div>
}
