/** Settings.
 *
 * There has been a `settings` view in the router since the beginning, but it
 * rendered the Engines panel, and three separate screens told people to change
 * something "in Settings" that they could not find. This is that screen: the
 * choices that are about the person rather than about the analysis, each one
 * saying in a sentence what it does and, where it matters, what it does not do.
 *
 * Nothing here changes a stored option, an estimator, or a row of data. That
 * separation is the whole reason presentation lives in its own screen.
 */

import { useEffect, useState } from 'react'
import { api } from '../api'
import { isDesktop, openExternal, pickFile } from '../desktop'
import { useStore } from '../store'
import type { Profile } from '../types'

type Theme = 'light' | 'dark' | 'system'

const THEMES: { id: Theme; label: string; blurb: string }[] = [
  { id: 'light', label: 'Light', blurb: 'Warm paper and dark ink. Best in a bright room, and what plots are exported in.' },
  { id: 'dark', label: 'Dark', blurb: 'Dark ground and light ink. Easier on the eyes at night and on a dim screen.' },
  { id: 'system', label: 'Match my computer', blurb: 'Follow the light or dark setting your operating system is using, and change with it.' },
]

const PROFILES: { id: Profile; label: string; blurb: string }[] = [
  {
    id: 'beginner',
    label: 'Guided',
    blurb: 'Fewer options on screen, and every step spelled out. Advanced settings are collapsed, not removed.',
  },
  {
    id: 'standard',
    label: 'Standard',
    blurb: 'The ordinary working view: the options most analyses need, and the rest one click away.',
  },
  {
    id: 'advanced',
    label: 'Advanced',
    blurb: 'Everything on screen at once, including the options that are usually best left alone.',
  },
]

export function Settings() {
  const theme = useStore((s) => s.theme)
  const setTheme = useStore((s) => s.setTheme)
  const profile = useStore((s) => s.profile)
  const setProfile = useStore((s) => s.setProfile)
  const mode = useStore((s) => s.mode)
  const setMode = useStore((s) => s.setMode)
  const setView = useStore((s) => s.setView)
  const setExplain = useStore((s) => s.setExplain)
  const engineDown = useStore((s) => s.engineDown)
  const toast = useStore((s) => s.toast)

  const [settings, setSettings] = useState<Record<string, unknown> | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (engineDown) return
    api.settings().then(setSettings).catch(() => setSettings(null))
  }, [engineDown])

  const save = async (patch: Record<string, unknown>) => {
    setSaving(true)
    try {
      setSettings(await api.saveSettings(patch))
      toast('success', 'Saved.')
    } catch (err) {
      toast('error', 'That setting could not be saved.',
        String((err as { detail?: string })?.detail ?? err))
    } finally {
      setSaving(false)
    }
  }

  const chooseInterpreter = async (key: 'python_path' | 'r_path', what: string) => {
    const picked = await pickFile([{ name: what, extensions: isWindows() ? ['exe'] : ['*'] }])
    if (picked) await save({ [key]: picked })
  }

  return (
    <div className="canvas-pad scroll settings">
      <header className="canvas-head">
        <h2>Settings</h2>
        <p className="hint">
          How Causal Capybara looks and where it keeps things. None of this changes an estimate: switching
          any of it never alters a stored option, an estimator, or a row of your data.
        </p>
      </header>

      <Section
        title="Appearance"
        blurb="Applies straight away, and is remembered for next time."
      >
        <fieldset className="choice-set">
          <legend className="panel-title">Colour theme</legend>
          <div className="choice-grid">
            {THEMES.map((t) => (
              <label key={t.id} className={`choice-card${theme === t.id ? ' on' : ''}`}>
                <input
                  type="radio"
                  name="capy-theme"
                  value={t.id}
                  checked={theme === t.id}
                  onChange={() => setTheme(t.id)}
                />
                <span className="choice-body">
                  <span className="choice-label">
                    <ThemeSwatch theme={t.id} />
                    {t.label}
                  </span>
                  <span className="choice-blurb">{t.blurb}</span>
                </span>
              </label>
            ))}
          </div>
          <p className="tiny hint">
            Plots follow the theme on screen. A plot you export or put in a report is always drawn on the
            light palette, so it stays legible on paper.
          </p>
        </fieldset>
      </Section>

      <Section
        title="How much to show"
        blurb="Two independent controls: how dense the screen is, and whether the app walks you through the steps."
      >
        <fieldset className="choice-set">
          <legend className="panel-title">Detail on screen</legend>
          <div className="choice-grid">
            {PROFILES.map((p) => (
              <label key={p.id} className={`choice-card${profile === p.id ? ' on' : ''}`}>
                <input
                  type="radio"
                  name="capy-profile"
                  value={p.id}
                  checked={profile === p.id}
                  onChange={() => setProfile(p.id)}
                />
                <span className="choice-body">
                  <span className="choice-label">{p.label}</span>
                  <span className="choice-blurb">{p.blurb}</span>
                </span>
              </label>
            ))}
          </div>
          <p className="tiny hint">
            This hides and shows controls. It never changes what a method does, and an option you set in
            Advanced stays set when you switch back.
          </p>
        </fieldset>

        <div className="setting-row">
          <label className="switch">
            <input
              type="checkbox"
              checked={mode === 'guided'}
              onChange={(e) => setMode(e.target.checked ? 'guided' : 'studio')}
            />
            <span>Show the step-by-step strip across the top</span>
          </label>
          <p className="tiny hint">
            The numbered route from a question to a result: design, roles, diagnose, methods, estimate,
            report. Turn it off once the route is familiar.
          </p>
        </div>
      </Section>

      <Section
        title="Engines"
        blurb="Causal Capybara computes with Python and, for some methods, R. Neither has to be installed for the app to open."
      >
        <p className="tiny hint">
          Engine health, what each method needs, and installing packages all live on their own screen.
        </p>
        <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
          <button className="btn" onClick={() => setView('engines')}>Open engine setup</button>
          {isDesktop() && (
            <>
              <button className="btn ghost" disabled={saving || engineDown}
                      onClick={() => void chooseInterpreter('python_path', 'Python interpreter')}>
                Choose my Python…
              </button>
              <button className="btn ghost" disabled={saving || engineDown}
                      onClick={() => void chooseInterpreter('r_path', 'Rscript')}>
                Choose my R…
              </button>
            </>
          )}
        </div>
        {settings && (
          <dl className="kv tiny" style={{ marginTop: 10 }}>
            {typeof settings.python_path === 'string' && settings.python_path && (
              <><dt>Python</dt><dd className="mono tiny">{settings.python_path}</dd></>
            )}
            {typeof settings.r_path === 'string' && settings.r_path && (
              <><dt>R</dt><dd className="mono tiny">{settings.r_path}</dd></>
            )}
          </dl>
        )}
      </Section>

      <Section
        title="Your data"
        blurb="Everything runs on this machine. Nothing is uploaded, and there is no telemetry of any kind."
      >
        {settings && typeof settings.home === 'string' && (
          <dl className="kv tiny">
            <dt>Projects and cache</dt><dd className="mono tiny">{settings.home}</dd>
          </dl>
        )}
        <div className="setting-row">
          <label className="switch">
            <input
              type="checkbox"
              checked={settings?.allow_downloads !== false}
              disabled={!settings || saving}
              onChange={(e) => void save({ allow_downloads: e.target.checked })}
            />
            <span>Let me download the published study datasets</span>
          </label>
          <p className="tiny hint">
            The ten real studies are fetched from their publisher only when you ask for one, checked
            against a recorded fingerprint, and kept on this machine. The request carries no project, no
            data and nothing that identifies you. Turn this off and nothing is ever downloaded.
          </p>
        </div>
      </Section>

      <Section title="Learn and about" blurb="Where to read more.">
        <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
          <button className="btn" onClick={() => setView('learn')}>Open the method catalogue</button>
          <button className="btn ghost" onClick={() => setExplain('concept.about')}>
            About Causal Capybara
          </button>
          <button
            className="btn ghost"
            onClick={() => void openExternal('https://github.com/HYHBalci/causal-capybara')}
          >
            Project page
          </button>
        </div>
      </Section>
    </div>
  )
}

function Section({ title, blurb, children }: {
  title: string
  blurb: string
  children: React.ReactNode
}) {
  return (
    <section className="card pad settings-section">
      <h3 className="settings-title">{title}</h3>
      <p className="hint settings-blurb">{blurb}</p>
      {children}
    </section>
  )
}

/** A two-tone chip so the choice can be recognised without reading it. */
function ThemeSwatch({ theme }: { theme: Theme }) {
  const pairs: Record<Theme, [string, string]> = {
    light: ['#faf7f2', '#1c1917'],
    dark: ['#1a1816', '#f2ede6'],
    system: ['#faf7f2', '#1a1816'],
  }
  const [a, b] = pairs[theme]
  return (
    <svg className="theme-swatch" width="22" height="14" viewBox="0 0 22 14" aria-hidden focusable="false">
      <rect x="0.5" y="0.5" width="21" height="13" rx="2.5" fill={a} stroke="var(--rule-strong)" />
      <path d="M11 0.5 H19 a2.5 2.5 0 0 1 2.5 2.5 v8 a2.5 2.5 0 0 1 -2.5 2.5 H11 Z" fill={b} />
    </svg>
  )
}

function isWindows(): boolean {
  return typeof navigator !== 'undefined' && /win/i.test(navigator.platform || navigator.userAgent)
}
