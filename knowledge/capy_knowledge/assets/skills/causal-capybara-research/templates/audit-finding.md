# Causal audit finding

Original template, Apache-2.0. Fill the fields from observed evidence; omit
irrelevant optional context. This template does not record approval or certify
identification.

- **Title and priority:** A concrete problem and its material effect on the
  requested analysis; explain the priority.
- **Scope and location:** Authorized repository revision and `path:line`, or an
  exact supplied-material section when no code was inspected.
- **Observation:** What the executed code or supplied evidence does. Include
  only the minimum relevant local excerpt; distinguish it from inference.
- **Threatened estimand or assumption:** State the population, contrast, scale
  or horizon, and the identifying/estimation condition affected.
- **Consequence:** Explain how the observed behavior changes the reported
  quantity or invalidates the intended calculation. Label design facts that
  cannot be determined from code as unresolved.
- **Correction or recommendation:** A scoped change preserving the intended
  target, or an explicit choice of target if the current question is ambiguous.
- **Evidence and citations:** Source IDs and URLs, available versions/locators,
  claim IDs, knowledge release/snapshot and applicable qualifications. Include
  counterevidence and coverage gaps. Do not imply that linked authors reviewed
  this implementation.
- **Verification status:** `inspected`, `tested`, or `unverified`, with exact
  performed checks, observed outcomes and remaining study evidence needed.
  Passing numerical checks does not establish causal identification.
- **Follow-through:** If implementation work was authorized and performed,
  report the change and relevant validation separately from the recommendation.
  Leave diagnostic acceptance, assumption acceptance and study approval to the
  applicable user workflow.
