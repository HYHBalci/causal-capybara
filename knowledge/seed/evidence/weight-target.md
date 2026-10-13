# Weighting and support are target-specific

SPDX-License-Identifier: Apache-2.0

Original Causal Capybara explanation; agent contextual review only. External publication bodies are not included.

ATE, ATT and overlap-population targets use different treatment weighting rules. Record the target before fitting assignment models, and examine support, weight concentration and balance after weighting. Clipping probabilities changes the estimator; discarding units can change the eligible target. Neither stable weights nor measured balance proves exchangeability, and no probability model repairs absent treatment support for the required comparison.

Source context: rosenbaum1983, bang2005.

This note does not certify a study, a complete implementation, or independent human review.
