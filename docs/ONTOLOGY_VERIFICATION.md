# Ontology implementation verification

Checked on 12 October 2026 against public `main` at `2f8fd3e`.
The implementation and release gates are described in
[ONTOLOGY_READINESS.md](ONTOLOGY_READINESS.md).

| Check | Observed result |
|---|---|
| Python integration | 418 passed, 7 Windows symlink privilege skips; assistant, adversarial privacy/approval/provenance, knowledge, publishing, benchmark, protocol, clients, readiness and project lifecycle |
| Frontend | 49 unit tests passed; TypeScript and production build passed |
| Browser | Research tree, context preview, proposals, failure/cancellation, revision handling and accessibility checks passed; a separate live-sidecar browser smoke exercised cited help and proposal staging; a real Save graph pointer click and toolbar containment passed in a 360 px graph pane |
| Pack compilation | Exact source hashes agree with Git bytes; same-runtime rebuild is byte-exact; 28 portability regressions reject altered canonical records, derivatives, SQLite/FTS contents and ranking statistics; bundled SQLite 3.50.4 verifies the shipped 3.49.1 index by strict semantic comparison |
| Public publishing | 167 tests passed, 6 Windows symlink skips; closed file/field contracts, all 16 file rights admissions, scoped review, purpose permissions and changed installed artifacts; independent review found no outstanding issues |
| Frozen retrieval evaluation | 30/30 answerable tasks found their expected claims in the top five; 10/10 unsupported tasks abstained; all returned citations resolved |
| Independent wheel | Installed wheel validates its bundled files and serves six read tools, three prompts and cited context over official SDK stdio |
| Actual Codex client | CLI 0.159.0 made successful context, requirements and evidence calls against the final installed wheel; scratch files remained unchanged |
| Bundled Windows runtime | Embedded Python 3.13.15 validates knowledge, imports the assistant and completes a synthetic staggered-DiD example with schema-valid results |
| Native application | Cargo checks/bootstrap test and Windows release compilation passed |
| Dependency notices | License collection covers the staged Python wheels, npm packages and Rust crates |

Final public knowledge snapshot:

```text
sha256:4786ac6d119d7e1168d7d21ab2515cd449fef68f1c1b0249a506e93bd7267f1a
```

The independently built `capy_knowledge-0.1.0-py3-none-any.whl` has SHA-256:

```text
0692e7668fd6a2af04d7a9f9a4356a72fa5a71d6ede55becb52d163ca275a88b
```

The numerical parity regression compares an approved assistant run with the
ordinary job engine using identical synthetic AIPW inputs. Reproduction checks
remain separate records with explicit tolerances and direct lineage; agreement
does not establish identification or scientific validity.

Automated checks are defined in [CI](../.github/workflows/ci.yml),
[the standalone ontology workflow](../.github/workflows/ontology.yml) and
[the Windows release workflow](../.github/workflows/release.yml). Local results
do not imply a particular GitHub Actions run passed; consult that run's output.

Release acceptance still requires independent human scientific review of the
bounded seed and benchmark, an actual skill-read/discovery result, real Claude
Code and Antigravity host checks, authenticated provider checks, and the native
approval interaction on a clean Windows installation. The Codex MCP query pass
does not satisfy its still-unverified skill-read gate. Setup templates are tested
without claiming that unavailable hosts were exercised. Broader design/engine
reproduction evaluations and upstream ancestry remain outside the checked scope.

No third-party paper bodies or personal overlays enter the public pack. Content
permissions are recorded per item; see [ONTOLOGY_RIGHTS.md](ONTOLOGY_RIGHTS.md).
