# Fix for rejection issue #2

## Problem

The original paper generated default through a nearly additive logistic index
and then treated logistic regression's strong performance as a substantive
negative result about nonlinear learners. That conclusion was structurally
favoured by the generator.

## Implemented correction

1. Preserved the original mechanism under the explicit name `additive_logit`.
2. Added threshold, conditional-interaction, and hybrid outcome regimes while
   keeping the covariate generator, feature set, noise scale, and late shock
   comparable.
3. Aligned expected prevalence in the challenge regimes by calibrating only the
   intercept.
4. Repeated the four-model comparison across ten complete regenerate/split/fit
   seeds for every regime.
5. Added deterministic tests for every regime and a loud failure for unknown
   regime names.
6. Rewrote the abstract, contribution list, model-benchmark section,
   discussion, threats to validity, and conclusion.

## Verified result

| Regime | Winning fixed configuration | Mean AUC | Logistic mean AUC |
|---|---|---:|---:|
| Additive-logit | Logistic regression | 0.6950 | 0.6950 |
| Threshold | Random forest | 0.7821 | 0.7658 |
| Conditional interaction | Histogram boosting | 0.7822 | 0.7557 |
| Hybrid | Extra trees | 0.6775 | 0.6770 |

The corrected conclusion is that learner ranking depends on alignment between
the learner and the declared outcome mechanism. The benchmark no longer claims
that representational complexity is generally ineffective.

## Remaining boundary

This fixes the structural circularity in the model-ranking claim by exposing
its sensitivity to multiple DGPs. It does not turn synthetic results into
evidence about real agricultural lending. It also does not solve the separate
review issue concerning hyperparameter tuning; all configurations remain fixed
and are described as such.
