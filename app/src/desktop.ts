/** The desktop shell, from the UI's side.
 *
 * Everything in here is optional. The same interface runs in a browser tab
 * during development, where none of it exists, so each function answers
 * honestly rather than throwing: `isDesktop()` is false, the file picker
 * returns null, and the caller falls back to the typed-path field.
 *
 * The Rust side has offered `engine_status` and `restart_engine` since the
 * first build, and the dialog and shell plugins have been registered and
 * permitted the whole time -- but nothing on this side could call them,
 * because the JavaScript half of the Tauri API was never a dependency. The
 * "Try again" button on the boot screen was therefore a guaranteed no-op:
 * nothing re-spawned the engine, so retrying could only fail again.
 */

interface TauriWindow {
  __TAURI_INTERNALS__?: unknown
}

/** True inside the packaged window, false in a browser tab. */
export function isDesktop(): boolean {
  return typeof window !== 'undefined' && !!(window as TauriWindow).__TAURI_INTERNALS__
}

/** Tauri's modules are only importable inside the window. Loading them eagerly
 *  would break `npm run dev` in a browser, so every entry point below imports
 *  on demand and treats a failure as "this capability is not here". */
async function mod<T>(loader: () => Promise<T>): Promise<T | null> {
  if (!isDesktop()) return null
  try {
    return await loader()
  } catch {
    return null
  }
}

export interface EngineState {
  running: boolean
  url: string
  detail: string
  /** The interpreter the shell last tried, so a message can name it. */
  python: string
}

export interface PythonProbe {
  ok: boolean
  path: string
  version: string
  missing: string[]
  detail: string
}

/** Ask whether a chosen interpreter can actually run the engine.
 *
 * Answering before the restart is the difference between "this one is missing
 * pandas and scipy" and watching a second silent failure. */
export async function probePython(path: string): Promise<PythonProbe | null> {
  const core = await mod(() => import('@tauri-apps/api/core'))
  if (!core) return null
  try {
    return await core.invoke<PythonProbe>('probe_python', { path })
  } catch {
    return null
  }
}

/** Remember an interpreter for good, so the next launch uses it too. */
export async function setPythonPath(path: string): Promise<string | null> {
  const core = await mod(() => import('@tauri-apps/api/core'))
  if (!core) return 'This only works in the installed app, not in a browser tab.'
  try {
    await core.invoke('set_python_path', { path })
    return null
  } catch (err) {
    return typeof err === 'string' ? err : String(err)
  }
}

/** Ask the shell whether the engine is answering, without going through HTTP.
 *  Useful when HTTP itself is what is failing. */
export async function engineStatus(): Promise<EngineState | null> {
  const core = await mod(() => import('@tauri-apps/api/core'))
  if (!core) return null
  try {
    return await core.invoke<EngineState>('engine_status')
  } catch {
    return null
  }
}

/** Restart the engine process without restarting the window.
 *
 * Returns the new state, or a sentence explaining why it could not start. The
 * sentence comes from the shell, which is the only part of the app that can
 * see the interpreter it tried and why it exited. */
export async function restartEngine(): Promise<{ ok: boolean; detail: string }> {
  const core = await mod(() => import('@tauri-apps/api/core'))
  if (!core) {
    return {
      ok: false,
      detail: 'Restarting the engine only works in the installed app, not in a browser tab.',
    }
  }
  try {
    const state = await core.invoke<EngineState>('restart_engine')
    return { ok: state.running, detail: state.detail }
  } catch (err) {
    return { ok: false, detail: typeof err === 'string' ? err : String(err) }
  }
}

export interface FileFilter {
  name: string
  extensions: string[]
}

/** Show the operating system's own file picker.
 *
 * Returns the chosen path, or null if the person cancelled or there is no
 * picker here. Typing an absolute path by hand was the only way into the app
 * before this, which is a full stop for anyone who has never used a terminal.
 */
export async function pickFile(filters?: FileFilter[]): Promise<string | null> {
  const dialog = await mod(() => import('@tauri-apps/plugin-dialog'))
  if (!dialog) return null
  try {
    const picked = await dialog.open({ multiple: false, directory: false, filters })
    if (picked === null) return null
    return typeof picked === 'string' ? picked : (picked as { path: string }).path
  } catch {
    return null
  }
}

/** Ask the operating system for a place to save a file. */
export async function pickSavePath(
  defaultName: string,
  filters?: FileFilter[],
): Promise<string | null> {
  const dialog = await mod(() => import('@tauri-apps/plugin-dialog'))
  if (!dialog) return null
  try {
    return await dialog.save({ defaultPath: defaultName, filters })
  } catch {
    return null
  }
}

/** Save exports through the native dialog in the desktop app. The dialog grants
 * filesystem access only to the selected path. Errors reach the caller so a
 * failed export cannot be mistaken for cancellation or a successful save. */
export async function saveFile(
  blob: Blob,
  defaultName: string,
  filters?: FileFilter[],
): Promise<string | null> {
  if (isDesktop()) {
    const { save } = await import('@tauri-apps/plugin-dialog')
    const path = await save({ defaultPath: defaultName, filters })
    if (path === null) return null
    const { writeFile } = await import('@tauri-apps/plugin-fs')
    await writeFile(path, new Uint8Array(await blob.arrayBuffer()))
    return path
  }
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = defaultName
  document.body.appendChild(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
  return defaultName
}

/** Open a URL in the person's own browser rather than inside the app window.
 *
 * Returns false when there is nowhere to open it, so the caller can show the
 * address instead of appearing to do nothing. */
export async function openExternal(url: string): Promise<boolean> {
  const shell = await mod(() => import('@tauri-apps/plugin-shell'))
  if (!shell) {
    if (typeof window !== 'undefined') {
      window.open(url, '_blank', 'noopener,noreferrer')
      return true
    }
    return false
  }
  try {
    await shell.open(url)
    return true
  } catch {
    return false
  }
}

/** Files dropped onto the window.
 *
 * `dragDropEnabled` has been true in the window config from the start, but
 * nothing listened, so dropping a dataset on the app did nothing at all.
 * Returns an unsubscribe function; a no-op outside the desktop. */
export async function onFilesDropped(
  handler: (paths: string[]) => void,
): Promise<() => void> {
  const webview = await mod(() => import('@tauri-apps/api/webview'))
  if (!webview) return () => {}
  try {
    const unlisten = await webview.getCurrentWebview().onDragDropEvent((event) => {
      if (event.payload.type === 'drop') handler(event.payload.paths)
    })
    return unlisten
  } catch {
    return () => {}
  }
}
