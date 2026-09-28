"""Concurrent first requests must not deadlock the reciprocal catalogues."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

# Force the historical lock inversion instead of relying on thread scheduling.
# The learner tries its lock while the explain request owns the registry lock.
# With separate locks it owns learn and waits for registry; explain then waits
# for learn forever. With one reentrant lock explain can finish first.
COLD_START = r'''
import json
import threading
from capy_sidecar import learn, registry

registry.refresh()
learn.refresh()
raw = learn._LOCK
attempted = threading.Event()
errors = []
results = {}

class ObservedLock:
    def __enter__(self):
        if threading.current_thread().name == "learn-request":
            if raw.acquire(blocking=False):
                attempted.set()
                return self
            attempted.set()
        raw.acquire()
        return self

    def __exit__(self, *_args):
        raw.release()

learn._LOCK = ObservedLock()

def load_learn():
    try:
        results["outline"] = learn.outline()
    except BaseException as exc:
        errors.append(repr(exc))

with registry._LOCK:
    worker = threading.Thread(target=load_learn, name="learn-request", daemon=True)
    worker.start()
    assert attempted.wait(5), "learning request did not reach its cache lock"
    results["explain"] = registry.explain_catalog()

worker.join(timeout=15)
assert not worker.is_alive(), "concurrent catalogue requests deadlocked"
assert not errors, errors
assert results["outline"] and results["explain"]
assert learn.entry("example", "lalonde")
assert learn.entry("study", "nsw_experimental")
print(json.dumps({"outline": len(results["outline"]), "explain": len(results["explain"])}))
'''


def test_concurrent_cold_catalogue_requests_complete(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "sidecar"), str(ROOT / "engines/python")])
    env["CAPY_HOME"] = str(tmp_path / "runtime")
    result = subprocess.run([sys.executable, "-c", COLD_START], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '"outline"' in result.stdout and '"explain"' in result.stdout
