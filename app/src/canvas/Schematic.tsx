/** The tiny schematics on the design cards, and the big ones on the boards.
 *
 * Drawn rather than illustrated: each one is the identification strategy, not
 * decoration. Same stroke weight, same accent, same restraint as the plots.
 */

const T = 'var(--teal)'
const S = 'var(--stone)'
const R = 'var(--rule-strong)'
const O = 'var(--ochre)'

export function Schematic({ kind, large = false }: { kind: string; large?: boolean }) {
  const size = large ? { width: 220, height: 120 } : { width: 92, height: 54 }
  const common = { ...size, viewBox: '0 0 220 120', fill: 'none', 'aria-hidden': true as const }
  switch (kind) {
    case 'urns':
      return (
        <svg {...common}>
          <path d="M40 30h34l-5 34H45z" stroke={S} strokeWidth="2" />
          <path d="M110 30h34l-5 34h-24z" stroke={T} strokeWidth="2" />
          <circle cx="57" cy="22" r="5" fill={S} />
          <circle cx="127" cy="22" r="5" fill={T} />
          <path d="M57 68v18h60v-18" stroke={R} strokeWidth="1.5" />
          <path d="M127 68v18" stroke={R} strokeWidth="1.5" />
          <rect x="70" y="88" width="66" height="20" rx="3" stroke={R} strokeWidth="1.5" />
          <text x="103" y="102" fontSize="11" fill="var(--muted)" textAnchor="middle">outcome</text>
        </svg>
      )
    case 'two_groups_confounders':
      return (
        <svg {...common}>
          <rect x="52" y="8" width="118" height="22" rx="4" stroke={R} strokeWidth="1.5" strokeDasharray="4 3" />
          <text x="111" y="23" fontSize="10" fill="var(--muted)" textAnchor="middle">measured confounders</text>
          <path d="M85 30v16M140 30v16" stroke={R} strokeWidth="1.5" markerEnd="url(#ar)" />
          <rect x="46" y="48" width="60" height="22" rx="4" stroke={S} strokeWidth="2" />
          <text x="76" y="63" fontSize="10" fill={S} textAnchor="middle">untreated</text>
          <rect x="118" y="48" width="56" height="22" rx="4" stroke={T} strokeWidth="2" />
          <text x="146" y="63" fontSize="10" fill={T} textAnchor="middle">treated</text>
          <path d="M76 70l24 20M146 70l-24 20" stroke={R} strokeWidth="1.5" />
          <rect x="80" y="90" width="62" height="20" rx="3" stroke={R} strokeWidth="1.5" />
          <text x="111" y="104" fontSize="10" fill="var(--muted)" textAnchor="middle">outcome</text>
          <Arrow />
        </svg>
      )
    case 'panel_grid':
      return (
        <svg {...common}>
          {[0, 1, 2, 3, 4].map((r) =>
            [0, 1, 2, 3, 4, 5, 6].map((c) => {
              const on = (r === 0 && c >= 2) || (r === 1 && c >= 4) || (r === 2 && c >= 5)
              return (
                <rect key={`${r}-${c}`} x={30 + c * 23} y={18 + r * 18} width="20" height="15" rx="2"
                      fill={on ? T : 'var(--paper-sunken)'} stroke={R} strokeWidth="1" />
              )
            }),
          )}
          <text x="110" y="115" fontSize="10" fill="var(--muted)" textAnchor="middle">
            units × periods, lit when treated
          </text>
        </svg>
      )
    case 'number_line_cutoff':
      return (
        <svg {...common}>
          <path d="M20 78h180" stroke={R} strokeWidth="1.5" />
          <path d="M110 20v70" stroke={O} strokeWidth="2.5" />
          {[26, 44, 62, 80, 98].map((x, i) => (
            <circle key={x} cx={x} cy={70 - i * 3} r="3.4" fill={S} />
          ))}
          {[122, 140, 158, 176, 194].map((x, i) => (
            <circle key={x} cx={x} cy={44 - i * 3} r="3.4" fill={T} />
          ))}
          <text x="110" y="16" fontSize="10" fill={O} textAnchor="middle">cutoff</text>
          <text x="110" y="95" fontSize="10" fill="var(--muted)" textAnchor="middle">running variable</text>
        </svg>
      )
    case 'iv_triangle':
      return (
        <svg {...common}>
          <circle cx="34" cy="60" r="15" stroke={T} strokeWidth="2" />
          <text x="34" y="64" fontSize="12" fill={T} textAnchor="middle">Z</text>
          <circle cx="110" cy="60" r="15" stroke={R} strokeWidth="2" />
          <text x="110" y="64" fontSize="12" fill="var(--ink)" textAnchor="middle">D</text>
          <circle cx="186" cy="60" r="15" stroke={R} strokeWidth="2" />
          <text x="186" y="64" fontSize="12" fill="var(--ink)" textAnchor="middle">Y</text>
          <path d="M49 60h46" stroke={T} strokeWidth="2" markerEnd="url(#ar)" />
          <path d="M125 60h46" stroke={R} strokeWidth="2" markerEnd="url(#ar)" />
          <path d="M42 44 Q110 4 178 44" stroke={O} strokeWidth="1.5" strokeDasharray="4 4" />
          <text x="110" y="16" fontSize="9" fill={O} textAnchor="middle">no arrow here — that is the assumption</text>
          <ellipse cx="148" cy="98" rx="20" ry="11" stroke={R} strokeWidth="1.2" strokeDasharray="3 3" />
          <text x="148" y="102" fontSize="9" fill="var(--muted)" textAnchor="middle">U</text>
          <path d="M140 88l-22-16M156 88l22-16" stroke={R} strokeWidth="1.2" markerEnd="url(#ar)" />
          <Arrow />
        </svg>
      )
    case 'one_vs_bundle':
      return (
        <svg {...common}>
          <path d="M20 88 Q52 70 84 74 T148 40 T200 26" stroke={T} strokeWidth="2.5" />
          {[0, 1, 2, 3, 4].map((i) => (
            <path key={i} d={`M20 ${86 + i * 3} Q60 ${74 + i * 4} 100 ${72 + i * 3} T200 ${64 + i * 5}`}
                  stroke={S} strokeWidth="1" opacity="0.5" />
          ))}
          <path d="M120 12v96" stroke={O} strokeWidth="2" strokeDasharray="5 4" />
          <text x="120" y="8" fontSize="9" fill={O} textAnchor="middle">intervention</text>
        </svg>
      )
    case 'line_with_event':
      return (
        <svg {...common}>
          <path d="M20 84 L60 78 L100 72 L140 40 L180 32 L200 28" stroke={T} strokeWidth="2.5" />
          <path d="M120 14v92" stroke={O} strokeWidth="2" strokeDasharray="5 4" />
          <path d="M120 66 L200 56" stroke={S} strokeWidth="1.5" strokeDasharray="3 3" />
          <text x="120" y="10" fontSize="9" fill={O} textAnchor="middle">interruption</text>
          <text x="176" y="70" fontSize="9" fill={S} textAnchor="middle">counterfactual</text>
        </svg>
      )
    case 'd_m_y':
      return (
        <svg {...common}>
          <circle cx="34" cy="60" r="15" stroke={R} strokeWidth="2" />
          <text x="34" y="64" fontSize="12" fill="var(--ink)" textAnchor="middle">D</text>
          <circle cx="110" cy="34" r="15" stroke={T} strokeWidth="2" />
          <text x="110" y="38" fontSize="12" fill={T} textAnchor="middle">M</text>
          <circle cx="186" cy="60" r="15" stroke={R} strokeWidth="2" />
          <text x="186" y="64" fontSize="12" fill="var(--ink)" textAnchor="middle">Y</text>
          <path d="M46 52l50-12M124 40l50 12" stroke={T} strokeWidth="2" markerEnd="url(#ar)" />
          <path d="M49 66h122" stroke={R} strokeWidth="2" markerEnd="url(#ar)" />
          <text x="110" y="88" fontSize="9" fill="var(--muted)" textAnchor="middle">direct path</text>
          <Arrow />
        </svg>
      )
    case 'person_time_grid':
      return (
        <svg {...common}>
          {[0, 1, 2].map((r) =>
            [0, 1, 2, 3, 4].map((c) => {
              const on = (r + c) % 3 === 0
              return (
                <rect key={`${r}-${c}`} x={40 + c * 30} y={26 + r * 24} width="24" height="18" rx="2"
                      fill={on ? T : 'var(--paper-sunken)'} stroke={R} strokeWidth="1" />
              )
            }),
          )}
          {[0, 1, 2, 3].map((c) => (
            <path key={c} d={`M${64 + c * 30} 20 q15 -8 30 0`} stroke={O} strokeWidth="1.4" markerEnd="url(#ar)" />
          ))}
          <text x="110" y="112" fontSize="9" fill="var(--muted)" textAnchor="middle">
            arrows cannot point into the past
          </text>
          <Arrow />
        </svg>
      )
    default:
      return (
        <svg {...common}>
          <rect x="60" y="40" width="100" height="40" rx="6" stroke={R} strokeWidth="1.5" strokeDasharray="4 4" />
        </svg>
      )
  }
}

function Arrow() {
  return (
    <defs>
      <marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0 0 L8 4 L0 8 z" fill="currentColor" />
      </marker>
    </defs>
  )
}
