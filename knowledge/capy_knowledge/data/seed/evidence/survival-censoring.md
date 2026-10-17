# Conditional censoring assumptions must match the procedure

SPDX-License-Identifier: Apache-2.0

Original Causal Capybara explanation; agent contextual review only. External publication bodies are not included.

A right-censored record says the event occurs after the observed duration; it contributes a survival-tail term rather than an event density to a parametric likelihood. Raw treatment-arm Kaplan–Meier and its RMST summary need within-arm independent censoring and supported risk sets for their associational distributions. Conditional independence of censoring given additional measured covariates does not automatically justify raw arm curves when both censoring and event time depend on those covariates. A causal treatment contrast additionally needs a separate identification argument. Delayed entry, interval censoring and competing events require explicitly supported procedures.

Source context: source:ontology:cole2004, source:ontology:wei1992, source:ontology:r-survreg.

This note does not certify a study, a complete implementation, or independent human review.
