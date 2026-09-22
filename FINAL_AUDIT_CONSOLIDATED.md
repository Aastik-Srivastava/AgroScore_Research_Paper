# AgroScore final audit — consolidated (post-merge)

This supersedes `FINAL_SUBMISSION_AUDIT.md` and `README_CORRECTIONS.md`, which
documented an earlier correction pass. That pass fixed real, independently
verified issues (see "Verified from the previous pass" below) but was built
from a snapshot of `agroscore.tex` that predated a second round of fixes.
This document reflects the merged, final state: the previous pass's fixes
plus the ones that had been dropped, re-applied and re-verified.

## Verified from the previous pass (independently re-checked in this session)

- **Outcome-regime redesign** (`agroscore_reproducible_study.py`,
  `regime_benchmark.py`): four outcome mechanisms (`additive_logit`,
  `threshold`, `interaction`, `hybrid`). Re-ran `regime_benchmark.py` from
  scratch — bit-for-bit identical to the numbers in `ISSUE_2_REDESIGN.md`
  and now in the manuscript. This is real evidence against the "generator
  structurally favours linear models" objection, not just hedging language.
- **DCR privacy metric fix** (shared scaler instead of two independent
  scalers): recomputed independently, got 0.9956, matches the manuscript.
- **`rasin2026conformance` citation**: verified real (arXiv:2606.08736,
  Muhammed Rasin, June 2026) via direct search — not a fabrication.
- **Leng et al. citation update** (volume 361, pp. 239–276, issue date June
  2026): verified against the live Springer page — accurate.
- **C2ST citation swap** (Lopez-Paz & Oquab 2017, replacing an unrelated
  health-records GAN citation for the detection-test methodology): correct
  and a genuine improvement.
- `test_agroscore.py` (7 tests): re-run, all pass.
- Reference/label/citation integrity: re-checked after all subsequent edits,
  0 missing refs, 0 duplicate labels, 0 missing or unused bib entries.

## Re-applied in this session (had been lost between drafts)

- ORCID (0009-0008-1146-1271) and Zenodo DOI
  (10.5281/zenodo.22871548) restored to the author footnote and the
  Code and Data Availability section.
- Regional parameter provenance restored to "ordinally anchored" to the
  Agriculture Census 2015–16 (`agcensus2016`, added back to
  `agroscore_additions.bib`), rather than "not estimated from any dataset."
- **Sample-Size Sensitivity section replaced.** The version in the uploaded
  zip had reverted to the single-seed, non-nested learning curve (correctly
  flagged in its own text as "descriptive rather than definitive") instead
  of the corrected design: a fixed 20,000-record common test population
  (seed 999999) with 8 independent training-seed replicates per sample size.
  The corrected design is restored, along with `fig_learning.pdf`,
  `learning_curve_v2.py`, and `results2/learning_curve_v2_{raw,summary}.csv`.
  This shows the full-vs-no-climate gap shrinking to ≤0.0011 AUC by
  n=32,000 with shrinking uncertainty — a real result, not a caveat.
- **Tuned-MLP fairness check restored** (`mlp_tune.py`,
  `results2/mlp_tuned.csv`): grid-search tuning raises MLP test AUC from
  0.5728 to 0.6090, closing roughly a third of the gap to logistic
  regression but not all of it. Added as Section VI-C in the manuscript.
- Cleaned up bibliography: removed the stale, now-superseded
  `leng2024satellite` entry (kept `leng2026satellite`, the corrected one)
  and the now-unused `sndv2018` entry (superseded by `lopezpaz2017c2st`).
- All 13 figure PDFs restored to `figures/` (only `fig_learning.pdf` needed
  regeneration; the other 12 are unchanged from the underlying
  `additive_logit`/seed-20260916 analysis and remain valid).

## Outstanding items — genuinely require the author's action, not further editing

1. **AI-assistance disclosure.** A previous pass noted "no disclosure text
   was added at the author's request." Check your target venue's current
   policy directly before submission; this is a compliance decision only
   you can make, and it varies by venue and can change between submission
   cycles.
2. **Freeze the Zenodo archive** so its contents are byte-identical to the
   code and seed that produced every number in this manuscript. A DOI that
   resolves to code producing different numbers than the paper is a
   reproducibility failure regardless of the DOI's validity.
3. **Full clean-environment run.** `python -m pip install -r requirements.txt
   && python run_all.py` has not been executed end-to-end with XGBoost
   installed inside this sandbox (XGBoost's wheel was unavailable in the
   earlier pass's restricted environment; it was available in this session's
   environment for the specific scripts re-run above, but `run_all.py` as a
   single entry point has not been exercised end-to-end here). Run it
   yourself in a clean environment before submission and diff the output
   against the numbers in the manuscript.
4. **Full PDF compile.** `IEEEtran.cls` was not reachable in either session's
   sandbox (restricted network). Compile in Overleaf, check for overfull
   boxes/clipped tables, and run the result through IEEE PDF eXpress.
5. **Venue selection and page-limit check.** ~16 journal-format pages.
6. **Double-blind rules**, if applicable, per venue.
7. **Official similarity check** (iThenticate/Turnitin via the venue) — no
   web-based check substitutes for this.

## Residual scientific limitations (disclosed in the manuscript itself)

All conclusions are conditional on author-specified synthetic outcome
processes; generator parameters are ordinally anchored but not fitted;
the outcome is partly circular by construction; hyperparameters are fixed
except where a tuning check is explicitly reported; the detector and DCR
tests are diagnostic screens with disclosed blind spots, not proofs of
distributional equivalence or memorisation-freedom; LOGO evaluation
isolates geography but leaks time; and no claim of real-portfolio fidelity,
external validity, or protected-group fairness is made anywhere. These are
disclosed as limitations, not omissions, and do not by themselves make the
benchmark unpublishable.
