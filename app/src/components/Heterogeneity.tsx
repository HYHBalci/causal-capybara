import type { Diagnostic, RunResult } from '../types'
import { ArtifactView, fmtNum, humanise } from './VegaChart'

/** These are direct tests of differences between treatment contrasts. Other
 * subgroup summaries do not become heterogeneity tests just by being shown. */
export function heterogeneityDiagnostics(result: RunResult): Diagnostic[] {
  return (result.diagnostics ?? []).filter((diagnostic) =>
    (diagnostic.id === 'heterogeneity' || diagnostic.id.endsWith('_heterogeneity')) &&
    diagnostic.values?.test_status !== undefined)
}

export function heterogeneityP(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Unavailable'
  if (value < 0.0001) return '< 0.0001'
  return value.toFixed(4)
}

export function heterogeneityScaleLabel(result: RunResult, diagnostic?: Diagnostic): string {
  const scale = diagnostic?.values?.scale ?? result.heterogeneity_target?.scale
  if (scale == null) return 'Not reported'
  if (result.heterogeneity_target?.exposure && (!diagnostic || diagnostic.id === 'outcome_heterogeneity')) {
    if (scale === 'mean_ratio') return 'Rate ratio (dimensionless)'
    if (scale === 'mean_difference') return 'Rate difference (counts per exposure unit)'
  }
  return humanise(String(scale))
}

export function HeterogeneityResults({ results }: { results: RunResult[] }) {
  return <>{results.filter((result) => result.status === 'ok').flatMap((result) =>
    heterogeneityDiagnostics(result).map((diagnostic) => {
      const values = diagnostic.values ?? {}
      const value = (key: string) => values[key] == null ? 'Not reported' : Array.isArray(values[key]) ? (values[key] as unknown[]).join(', ') : String(values[key])
      const artifacts = (result.artifacts ?? []).filter((artifact) => (diagnostic.artifact_ids ?? []).includes(artifact.id))
      return <section key={`${result.run_id}-${diagnostic.id}`} className="learn-block heterogeneity-results" aria-label={`Heterogeneity: ${result.method_label ?? result.method}`}>
        <h2>{diagnostic.title} · {result.method_label ?? result.method}</h2>
        <p>{diagnostic.summary}</p>
        <dl className="kv">
          <dt>Baseline modifier</dt><dd>{value('modifier')}</dd>
          <dt>Groups</dt><dd>{value('levels')}</dd>
          <dt>Reference group or value</dt><dd>{value('reference_level')}</dd>
          <dt>Contrast scale</dt><dd>{heterogeneityScaleLabel(result, diagnostic)}</dd>
          {(values.outcome_family != null || values.family != null) && <><dt>Outcome family</dt><dd>{humanise(String(values.outcome_family ?? values.family))}</dd></>}
          {values.horizon != null && <><dt>Follow-up horizon</dt><dd>{value('horizon')} {value('time_unit')}</dd></>}
          <dt>Interpretation</dt><dd>{values.interpretation == null ? result.interpretation ?? 'See the method assumptions' : value('interpretation')}</dd>
          {result.heterogeneity_target && <><dt>Target population</dt><dd>{result.heterogeneity_target.population}</dd>
            <dt>Inference target</dt><dd>{result.heterogeneity_target.inference_target}</dd></>}
          {result.heterogeneity_target?.exposure && <><dt>Fixed exposure</dt><dd>{result.heterogeneity_target.exposure}; predictions at one exposure unit. Rate ratios are dimensionless; rate differences are counts per exposure unit.</dd></>}
          <dt>Direct heterogeneity test</dt><dd>{value('test')}</dd>
          <dt>Test status</dt><dd>{humanise(value('test_status'))}</dd>
          <dt>Omnibus p-value</dt><dd>{['ok', 'estimated'].includes(String(values.test_status)) ? heterogeneityP(values.p_value) : 'Unavailable'}</dd>
          {typeof values.statistic === 'number' && <><dt>Test statistic</dt><dd>{fmtNum(values.statistic)}</dd></>}
          {values.df != null && <><dt>Degrees of freedom</dt><dd>{value('df')}</dd></>}
          {values.unavailable_reason != null && <><dt>Why unavailable</dt><dd>{value('unavailable_reason')}</dd></>}
          <dt>Multiplicity and intervals</dt><dd>{value('multiplicity')}</dd>
          {typeof values.ci_level === 'number' && <><dt>Pointwise confidence level</dt><dd>{Math.round(values.ci_level * 100)}%</dd></>}
          {values.bootstrap_usable != null && <><dt>Bootstrap fits</dt><dd>{value('bootstrap_usable')} usable of {value('bootstrap_reps')}; {value('bootstrap_failures')} failed</dd></>}
          <dt>Subgroups prespecified</dt><dd>{values.prespecified === true ? 'Declared by the analyst' : 'Not declared'}</dd>
        </dl>
        <p className="hint">Compare treatment contrasts directly on the stated scale. A small p-value in one group and a larger p-value in another do not establish a difference between groups. A large heterogeneity p-value does not establish equal effects. Inspect support, uncertainty and the planned family of tests.</p>
        {artifacts.length > 0 && <details>
          <summary>Subgroup estimates, direct comparisons and uncertainty</summary>
          {artifacts.map((artifact) => <ArtifactView key={artifact.id} artifact={artifact} height={280} />)}
        </details>}
      </section>
    }))}</>
}

/** Selected runs can describe different outcomes and different causal targets.
 * Separate their scales before drawing a forest or summarising their spread. */
export function comparableResultGroups(results: RunResult[]): RunResult[][] {
  const groups = new Map<string, RunResult[]>()
  for (const result of results) {
    const target = result.survival_target
    const heterogeneity = heterogeneityDiagnostics(result)[0]?.values
    const key = JSON.stringify([
      result.spec_id, result.design, result.estimand, result.outcome, result.treatment,
      result.interpretation ?? null, target?.scale ?? null, target?.time_unit ?? null,
      target?.horizon ?? null, target?.population ?? null, target?.contrast ?? null,
      heterogeneity?.outcome_family ?? heterogeneity?.family ?? null, heterogeneity?.scale ?? null,
      result.heterogeneity_target ?? null,
    ])
    const group = groups.get(key) ?? []
    group.push(result)
    groups.set(key, group)
  }
  return [...groups.values()]
}
