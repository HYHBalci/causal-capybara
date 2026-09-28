/** The last line of defence against a blank window.
 *
 * A render-time throw in React unmounts the whole tree. In a browser tab you at
 * least get a console; in a packaged desktop app you get an empty window and no
 * way to find out why -- which is exactly how the API-base bug presented, as a
 * black screen with the answer only reachable over a remote debugging port.
 *
 * Whatever else breaks, this shows the error, says the likely cause, and gives
 * a way out.
 */

import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
  stack: string | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, stack: null }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Still log it: a developer with devtools open should see the real trace.
    console.error('Causal Capybara failed to render', error, info)
    this.setState({ stack: info.componentStack ?? null })
  }

  render(): ReactNode {
    const { error, stack } = this.state
    if (!error) return this.props.children

    return (
      <div className="crash">
        <div className="crash-card card">
          <h1>Causal Capybara stopped drawing</h1>
          <p>
            Something threw while rendering, so the window is showing this instead of nothing at all.
            Your projects are folders on disk and are unaffected — nothing here has touched them.
          </p>
          <p className="hint">
            The most common cause is the engine not answering. It listens on
            <code> 127.0.0.1:8760</code>; if it is not running, start it and reload.
          </p>
          <pre className="crash-detail">{error.message}</pre>
          {stack && (
            <details>
              <summary>Where it happened</summary>
              <pre className="crash-detail">{stack.trim()}</pre>
            </details>
          )}
          <div className="row" style={{ gap: 8, marginTop: 14 }}>
            <button className="btn primary" onClick={() => window.location.reload()}>
              Reload
            </button>
            <button
              className="btn"
              onClick={() => {
                const text = `${error.message}\n\n${stack ?? ''}`.trim()
                void navigator.clipboard?.writeText(text)
              }}
            >
              Copy the error
            </button>
          </div>
        </div>
      </div>
    )
  }
}
