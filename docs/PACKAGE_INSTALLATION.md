# Package installation and recovery

The Engine setup screen starts an installation only after the user approves a
package plan or presses the explicit core-package repair button. It then tracks
a saved operation, showing elapsed time and live installer output. Returning to
the screen or reloading reconnects to that operation.

This resolves [issue #3](https://github.com/HYHBalci/causal-capybara/issues/3).
Previously the screen waited for one HTTP response for up to 120 seconds while
the installer could run for 900 seconds. A request timeout incorrectly cleared
the busy state even though the package process could still be running.

## Operation lifecycle

- `POST /engines/install` without literal `approved: true` returns a plan and
  starts no installation. An approved request returns a queued operation promptly.
- `GET /engines/installs` and `GET /engines/installs/{id}` return saved status,
  bounded output and elapsed time. Status requests never start package processes.
- The UI saves a request ID before sending approval. Retrying that ID returns the
  same operation, including after completion. If another package group is already
  running, the server returns that active operation and the UI follows it.
- A dropped connection keeps the pending request and disables new installations.
  The screen polls for its saved operation. If the server confirms no such record
  exists, the user can explicitly retry the same request ID.
- Results distinguish installed, partial, missing, failed and interrupted states.
  Package health is checked after the process exits; an HTTP timeout cannot be
  reported as installation success or failure.
- After a sidecar restart, a recorded process is checked by PID and birth identity.
  A live or inaccessible process keeps the installation lock. An exited process
  with no saved result is reported as interrupted and requires a package-health
  check. Recovery never terminates a PID loaded from disk.

Operations are stored atomically under `CAPY_HOME/installations`. Only one
installation started through a sidecar instance can be active at a time. The
synchronous command-line installer remains available; an independently launched
external package manager is outside this app-level coordination.

## Regression coverage

Backend tests cover prompt HTTP responses, explicit approval, duplicate and
cross-package requests, live output, bounded logs, persisted/restarted operations,
orphan-process identity, unknown outcomes and stdout closing before process exit.
Frontend tests cover failed acknowledgements, polling, reload recovery, missing
records, explicit retries and a competing package operation. The hosted browser
readiness suite checks the approval and progress controls using a controlled
installer response; it downloads no packages.

```sh
python -m pytest tests/python/test_engine_install_jobs.py tests/python/test_engine_install.py tests/python/test_readiness.py -q
npm --prefix app test
npm --prefix app run test:ui
```

Keep the desktop app open while installing. If it closes unexpectedly, reopen
Engine setup and use **Check installation status** before taking another action.
