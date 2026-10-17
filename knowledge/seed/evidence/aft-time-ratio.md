# Ordinary AFT regression and structural failure-time models differ

SPDX-License-Identifier: Apache-2.0

Original Causal Capybara explanation; agent contextual review only. External publication bodies are not included.

A common-distribution AFT model relates log event time to covariates; exponentiating a treatment coefficient gives a conditional time factor under that specification. It is not automatically a causal parameter or a marginal RMST contrast. Declare the distribution, shape, scale and covariate parameterization, since software conventions differ. Structural failure-time and rank-preserving models encode separate counterfactual transformations and identifying conditions; an ordinary AFT fit does not implement them.

Source context: source:ontology:wei1992, source:ontology:r-survreg, source:ontology:lifelines-aft, source:ontology:robins1991.

This note does not certify a study, a complete implementation, or independent human review.
