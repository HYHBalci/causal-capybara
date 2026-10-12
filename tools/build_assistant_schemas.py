"""Reproduce strict assistant request contracts without provider dependencies."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sidecar"))
from capy_sidecar.ontology_api import (ApprovalRequest, ChatRequest, ContextRequest,
    ExecutionRequest, KnowledgeContextRequest, ProposalRequest, ReproductionRequest)

for name, model in {"context.request": ContextRequest, "chat.request": ChatRequest,
                    "proposal.request": ProposalRequest, "approval.request": ApprovalRequest,
                    "execution.request": ExecutionRequest, "knowledge-context.request": KnowledgeContextRequest,
                    "reproduction.request": ReproductionRequest}.items():
    schema = model.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://causalcapybara.dev/schemas/capy.assistant.{name}.v1.json"
    (ROOT / "schemas" / f"capy.assistant.{name}.v1.json").write_text(json.dumps(schema, sort_keys=True, indent=2) + "\n", encoding="utf-8")
