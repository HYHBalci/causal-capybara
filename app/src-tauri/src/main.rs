// Causal Capybara desktop shell.
//
// The window is the product; this file is the plumbing. Its jobs are:
//   * start the Python sidecar as a child process and wait until it answers,
//   * shut it down cleanly when the window closes, so a killed app does not
//     leave an orphaned engine holding a port and a project lock,
//   * hand the UI file dialogs, because a research tool that cannot open a file
//     is not one.
//
// The desktop never imports pandas or MatchIt. It sends a spec and receives a
// result -- everything statistical happens behind the HTTP boundary.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, State};

const SIDECAR_PORT: u16 = 8760;
const SIDECAR_HOST: &str = "127.0.0.1";
const STARTUP_BUDGET: Duration = Duration::from_secs(45);

struct Sidecar(Mutex<Option<Child>>);

/// Why the engine is not running, in the words of the process that tried to
/// start it.
///
/// This used to go to `eprintln!` only. The release build is compiled with
/// `windows_subsystem = "windows"`, so it has no console: the single most
/// useful sentence in the application -- which interpreter was tried and what
/// it failed to import -- was written to a handle that does not exist, and the
/// window showed a generic "not answering" instead.
struct LastError(Mutex<Option<String>>);

#[derive(serde::Serialize)]
struct EngineStatus {
    running: bool,
    url: String,
    detail: String,
    /// The interpreter the last spawn attempt used, so the recovery screen can
    /// name it rather than talking about "Python" in the abstract.
    python: String,
}

#[derive(serde::Serialize)]
struct PythonProbe {
    ok: bool,
    path: String,
    version: String,
    missing: Vec<String>,
    detail: String,
}

/// The interpreter the person chose, remembered between runs.
///
/// `CAPY_PYTHON` still wins, because somebody who sets an environment variable
/// knows something we do not. This file is what the "point me at my Python"
/// button writes, and it lives in the platform's own config directory so it
/// survives reinstalling the app.
fn engine_config_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|d| d.join("engine.json"))
}

fn saved_python(app: &tauri::AppHandle) -> Option<(String, bool)> {
    let path = engine_config_path(app)?;
    // Keep an existing interpreter selection after correcting the app identifier.
    // Old preview selections remain fallbacks; new selections explicitly
    // override the bundled interpreter for recovery if the bundle is broken.
    let legacy = path
        .parent()?
        .parent()?
        .join("dev.vcesr.casualcapybara")
        .join("engine.json");
    let text = std::fs::read_to_string(&path)
        .or_else(|_| std::fs::read_to_string(legacy))
        .ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let python = value.get("python")?.as_str()?.trim().to_string();
    if python.is_empty() || !PathBuf::from(&python).is_file() {
        return None;
    }
    let override_bundled = value
        .get("override_bundled")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    Some((python, override_bundled))
}

/// Where the Python engine lives, whether we are running from a checkout or
/// from an installed bundle.
fn resolve_paths(app: &tauri::AppHandle) -> Option<(PathBuf, PathBuf, PathBuf)> {
    // installed bundle: resources/{sidecar,engines}
    if let Ok(res) = app.path().resource_dir() {
        let sidecar = res.join("sidecar");
        if sidecar.is_dir() {
            return Some((res.clone(), sidecar, res.join("engines").join("python")));
        }
    }
    // development: walk up from the executable to the repository root
    let mut here = std::env::current_dir().ok()?;
    for _ in 0..6 {
        if here.join("sidecar").is_dir() && here.join("engines").is_dir() {
            return Some((
                here.clone(),
                here.join("sidecar"),
                here.join("engines").join("python"),
            ));
        }
        here = here.parent()?.to_path_buf();
    }
    None
}

/// The interpreter to run the sidecar with. A bundled runtime wins; otherwise
/// we fall back to whatever python is on the path, and the UI shows the engine
/// as unavailable if that fails rather than refusing to start.
/// Find a Python that can actually run the engine.
///
/// Order matters. An explicit CAPY_PYTHON wins over everything, because the
/// person who set it knows something we do not. Then a managed runtime beside
/// the app, then a virtualenv -- searched up the tree as well as at the root,
/// since in an installed build `root` is the bundle directory and a developer's
/// .venv sits several levels above it. The bare interpreter on PATH is the last
/// resort and usually the wrong one: it exists, so it is chosen, and then it
/// turns out to have none of the dependencies.
fn python_executable(app: &tauri::AppHandle, root: &PathBuf) -> String {
    if let Ok(explicit) = std::env::var("CAPY_PYTHON") {
        if !explicit.trim().is_empty() {
            return explicit;
        }
    }
    let (managed, venv) = if cfg!(windows) {
        ("runtimes/python/python.exe", ".venv/Scripts/python.exe")
    } else {
        ("runtimes/python/bin/python3", ".venv/bin/python")
    };
    let saved = saved_python(app);
    if let Some((chosen, true)) = saved.as_ref() {
        return chosen.clone();
    }
    // An installed release starts with its tested runtime, including after a
    // preview saved an external Python. The recovery screen may override it.
    let bundled = root.join(managed);
    if bundled.is_file() {
        return bundled.to_string_lossy().into_owned();
    }
    if let Some((chosen, _)) = saved {
        return chosen;
    }

    let mut candidates: Vec<PathBuf> = vec![root.join(managed), root.join(venv)];
    let mut dir = root.as_path();
    for _ in 0..5 {
        match dir.parent() {
            Some(parent) => {
                candidates.push(parent.join(managed));
                candidates.push(parent.join(venv));
                dir = parent;
            }
            None => break,
        }
    }

    for c in candidates {
        if c.is_file() {
            return c.to_string_lossy().into_owned();
        }
    }
    if cfg!(windows) {
        "python".into()
    } else {
        "python3".into()
    }
}

fn valid_health_response(response: &str) -> bool {
    let Some((headers, body)) = response.split_once("\r\n\r\n") else {
        return false;
    };
    if !matches!(
        headers.lines().next(),
        Some("HTTP/1.1 200 OK" | "HTTP/1.0 200 OK")
    ) {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(body)
        .map(|health| {
            health["ok"] == true
                && health["app"] == "Causal Capybara"
                && health["schema_version"] == 1
        })
        .unwrap_or(false)
}

fn sidecar_alive() -> bool {
    let Ok(mut socket) = std::net::TcpStream::connect_timeout(
        &format!("{SIDECAR_HOST}:{SIDECAR_PORT}").parse().unwrap(),
        Duration::from_millis(400),
    ) else {
        return false;
    };
    let timeout = Some(Duration::from_millis(600));
    let _ = socket.set_read_timeout(timeout);
    let _ = socket.set_write_timeout(timeout);
    if socket
        .write_all(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1:8760\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut response = String::new();
    if socket.take(16_384).read_to_string(&mut response).is_err() {
        return false;
    }
    valid_health_response(&response)
}

fn spawn_sidecar(app: &tauri::AppHandle) -> Result<Option<Child>, String> {
    if sidecar_alive() {
        // A developer already has one running; leave it alone rather than
        // fighting it for the port.
        return Ok(None);
    }
    let (root, sidecar_dir, engine_dir) = resolve_paths(app).ok_or_else(|| {
        "This copy of Causal Capybara is missing the files it computes with. Reinstalling the app \
         should replace them."
            .to_string()
    })?;
    let python = python_executable(app, &root);

    let sep = if cfg!(windows) { ";" } else { ":" };
    let pythonpath = format!(
        "{}{}{}",
        engine_dir.to_string_lossy(),
        sep,
        sidecar_dir.to_string_lossy()
    );

    let mut cmd = Command::new(&python);
    // Keep the installed resource tree read-only and removable by setup.
    // Embedded Python ignores PYTHONDONTWRITEBYTECODE, so pass -B explicitly.
    cmd.arg("-B")
        .arg("-m")
        .arg("capy_sidecar")
        .arg("serve")
        .arg("--port")
        .arg(SIDECAR_PORT.to_string())
        .current_dir(&root)
        .env("PYTHONPATH", pythonpath)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd.spawn().map_err(|e| {
        format!(
            "Causal Capybara could not start Python. It tried to run '{python}', and the system \
             said: {e}. If Python is not installed on this computer, install it and open Causal \
             Capybara again; if it is, point the app at it from this screen."
        )
    })?;

    // Drain the engine's output into the app log rather than letting the pipe
    // fill and block the child. The last few lines are also kept in memory,
    // because when the engine dies during startup they contain the only
    // explanation there is, and the packaged build has no console to print to.
    let tail = std::sync::Arc::new(Mutex::new(Vec::<String>::new()));
    if let Some(out) = child.stderr.take() {
        let tail = tail.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                eprintln!("[engine] {line}");
                if let Ok(mut buf) = tail.lock() {
                    buf.push(line);
                    if buf.len() > 40 {
                        buf.remove(0);
                    }
                }
            }
        });
    }
    if let Some(out) = child.stdout.take() {
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                println!("[engine] {line}");
            }
        });
    }

    let deadline = Instant::now() + STARTUP_BUDGET;
    while Instant::now() < deadline {
        if sidecar_alive() {
            return Ok(Some(child));
        }
        if let Ok(Some(_status)) = child.try_wait() {
            let reason = tail
                .lock()
                .ok()
                .and_then(|buf| {
                    buf.iter()
                        .rev()
                        .find(|l| l.contains("Error") || l.contains("error"))
                        .cloned()
                })
                .unwrap_or_default();
            let because = if reason.is_empty() {
                String::new()
            } else {
                format!(" It stopped with: {reason}.")
            };
            return Err(format!(
                "Causal Capybara found Python at '{python}', but that copy is missing the packages \
                 it needs to compute with.{because} The usual fix is to point the app at a \
                 different Python, or to install the packages into this one."
            ));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    // A failed startup must not leave an untracked engine holding the port.
    let _ = child.kill();
    let _ = child.wait();
    Err(format!(
        "Python started from '{python}' but never became ready. Something may be blocking the \
         local connection the app uses to talk to it -- a firewall prompt waiting for an answer, \
         or another program already using port {SIDECAR_PORT}."
    ))
}

#[tauri::command]
fn engine_status(app: tauri::AppHandle, last: State<LastError>) -> EngineStatus {
    let running = sidecar_alive();
    let python = resolve_paths(&app)
        .map(|(root, _, _)| python_executable(&app, &root))
        .unwrap_or_else(|| "python".into());
    let detail = if running {
        "Causal Capybara is ready to compute.".into()
    } else {
        // The recorded reason is the whole point: without it the window can
        // only say "not answering", which tells nobody anything.
        last.0
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_else(|| {
                "Causal Capybara cannot reach the part of itself that does the computing. \
                 Everything you can read is still here; estimating is not."
                    .into()
            })
    };
    EngineStatus {
        running,
        url: format!("http://{SIDECAR_HOST}:{SIDECAR_PORT}"),
        detail,
        python,
    }
}

/// Check an interpreter before committing to it.
///
/// A person who has just been asked to "find your Python" deserves a straight
/// answer about whether the file they picked will work, rather than watching a
/// restart fail for reasons they cannot see.
#[tauri::command]
fn probe_python(path: String) -> PythonProbe {
    let script = "import sys, importlib.util as u; \
                  mods=['fastapi','uvicorn','pandas','numpy','scipy','yaml','pydantic',\
                        'pyarrow','sklearn','statsmodels','linearmodels','jsonschema']; \
                  print(sys.version.split()[0]); \
                  print(','.join(m for m in mods if u.find_spec(m) is None))";
    let mut cmd = Command::new(&path);
    cmd.arg("-c").arg(script);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    match cmd.output() {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let mut lines = text.lines();
            let version = lines.next().unwrap_or("").trim().to_string();
            let missing: Vec<String> = lines
                .next()
                .unwrap_or("")
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let mut parts = version
                .split('.')
                .filter_map(|part| part.parse::<u32>().ok());
            let supported =
                parts.next() == Some(3) && parts.next().map_or(false, |minor| minor >= 11);
            let ok = supported && missing.is_empty();
            let detail = if !supported {
                format!("Python {version} is not supported. Choose Python 3.11 or newer in the Python 3 series.")
            } else if ok {
                format!("Python {version} here has everything Causal Capybara needs.")
            } else {
                format!(
                    "This is Python {version}, but it is missing {}. Install the app's requirements \
                     into this interpreter, or choose a Python environment that already has them.",
                    missing.join(", ")
                )
            };
            PythonProbe {
                ok,
                path,
                version,
                missing,
                detail,
            }
        }
        Ok(out) => PythonProbe {
            ok: false,
            path,
            version: String::new(),
            missing: vec![],
            detail: format!(
                "That file did not behave like a Python interpreter: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        },
        Err(err) => PythonProbe {
            ok: false,
            path,
            version: String::new(),
            missing: vec![],
            detail: format!("That file could not be run: {err}"),
        },
    }
}

/// Remember the interpreter the person chose, so it is used from now on.
#[tauri::command]
fn set_python_path(app: tauri::AppHandle, path: String) -> Result<(), String> {
    let target = engine_config_path(&app).ok_or_else(|| {
        "Causal Capybara has nowhere to save this choice on this computer.".to_string()
    })?;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            format!(
                "That choice could not be saved to {}: {e}",
                parent.display()
            )
        })?;
    }
    let body = serde_json::json!({ "python": path, "override_bundled": true }).to_string();
    std::fs::write(&target, body).map_err(|e| {
        format!(
            "That choice could not be saved to {}: {e}",
            target.display()
        )
    })
}

/// Restart the engine after a crash, or after being pointed at a new Python.
#[tauri::command]
fn restart_engine(
    app: tauri::AppHandle,
    state: State<Sidecar>,
    last: State<LastError>,
) -> Result<EngineStatus, String> {
    {
        let mut guard = state.0.lock().map_err(|e| e.to_string())?;
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    match spawn_sidecar(&app) {
        Ok(child) => {
            *state.0.lock().map_err(|e| e.to_string())? = child;
            if let Ok(mut g) = last.0.lock() {
                *g = None;
            }
            Ok(engine_status(app.clone(), last))
        }
        Err(message) => {
            // Keep the reason so the screen can show it even after this call
            // returns, then hand the same sentence back to the caller.
            if let Ok(mut g) = last.0.lock() {
                *g = Some(message.clone());
            }
            Err(message)
        }
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(Sidecar(Mutex::new(None)))
        .manage(LastError(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            engine_status,
            restart_engine,
            probe_python,
            set_python_path
        ])
        .setup(|app| {
            // Fit the window to the screen it is opening on.
            //
            // A fixed default size is a guess about somebody else's monitor. At
            // 150% display scaling a 1920x1080 screen is 1280x720 in the units
            // this window is measured in, so a window asking for more than that
            // opens with its own controls off the bottom of the desktop -- and
            // the person cannot resize it back, because the parts you drag are
            // the parts that are off-screen.
            if let Some(window) = app.get_webview_window("main") {
                if let Ok(Some(monitor)) = window.current_monitor() {
                    let scale = monitor.scale_factor();
                    let screen = monitor.size().to_logical::<f64>(scale);
                    // Leave room for the taskbar or dock, which the monitor
                    // size includes but the usable area does not.
                    let max_w = (screen.width - 40.0).max(900.0);
                    let max_h = (screen.height - 80.0).max(600.0);
                    if let Ok(current) = window.inner_size() {
                        let now = current.to_logical::<f64>(scale);
                        if now.width > max_w || now.height > max_h {
                            let _ = window.set_size(tauri::LogicalSize::new(
                                now.width.min(max_w),
                                now.height.min(max_h),
                            ));
                            let _ = window.center();
                        }
                    }
                }
            }

            // Never fail startup because an engine is missing. Open the window,
            // let the UI show the engine as unavailable, and disable Estimate.
            match spawn_sidecar(app.handle()) {
                Ok(child) => {
                    let state: State<Sidecar> = app.state();
                    *state.0.lock().unwrap() = child;
                }
                Err(message) => {
                    eprintln!("[capy] {message}");
                    // The window is about to ask what went wrong, and this is
                    // the only place that knows. Bind the lock before matching,
                    // for the same reason the exit handler below does.
                    let last: State<LastError> = app.state();
                    let locked = last.0.lock();
                    if let Ok(mut g) = locked {
                        *g = Some(message);
                    }
                }
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build Causal Capybara")
        .run(|app, event| {
            if let RunEvent::ExitRequested { .. } | RunEvent::Exit = event {
                let state: State<Sidecar> = app.state();
                // Bind the lock before matching on it. In an `if let`, the
                // temporary returned by lock() lives to the end of the block,
                // which is after `state` itself is dropped -- so the borrow
                // outlives what it borrows from and this does not compile.
                let locked = state.0.lock();
                if let Ok(mut guard) = locked {
                    if let Some(mut child) = guard.take() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::valid_health_response;

    #[test]
    fn only_accepts_the_engine_health_protocol() {
        assert!(valid_health_response("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"ok\":true,\"app\":\"Causal Capybara\",\"schema_version\":1}"));
        assert!(!valid_health_response(
            "HTTP/1.1 200 OK\r\n\r\n{\"ok\":true,\"app\":\"Other app\",\"schema_version\":1}"
        ));
        assert!(!valid_health_response(
            "HTTP/1.1 200 OK\r\n\r\n{\"ok\":true,\"app\":\"Causal Capybara\",\"schema_version\":2}"
        ));
        assert!(!valid_health_response("HTTP/1.1 503 Service Unavailable\r\n\r\n{\"ok\":true,\"app\":\"Causal Capybara\",\"schema_version\":1}"));
        assert!(!valid_health_response(
            "HTTP/1.1 200 OK\r\n\r\n<html>Another service</html>"
        ));
    }
}
