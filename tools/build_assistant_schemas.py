"""Reproduce strict assistant request contracts without provider dependencies."""
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sidecar"))
from capy_sidecar.ontology_api import (ApprovalRequest, ChatRequest, ContextRequest,
    ExecutionRequest, HandoffRequest, KnowledgeContextRequest, ProposalRequest, ReproductionRequest)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
stale = []
for name, model in {"context.request": ContextRequest, "chat.request": ChatRequest,
                    "proposal.request": ProposalRequest, "approval.request": ApprovalRequest,
                    "execution.request": ExecutionRequest, "knowledge-context.request": KnowledgeContextRequest,
                    "reproduction.request": ReproductionRequest, "handoff.request": HandoffRequest}.items():
    schema = model.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://causalcapybara.dev/schemas/capy.assistant.{name}.v1.json"
    path = ROOT / "schemas" / f"capy.assistant.{name}.v1.json"
    content = (json.dumps(schema, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if args.check:
        if not path.is_file() or path.read_bytes().replace(b"\r\n", b"\n") != content:
            stale.append(path.name)
    else:
        path.write_bytes(content)
if stale:
    raise SystemExit("Assistant schemas are stale: " + ", ".join(stale))
