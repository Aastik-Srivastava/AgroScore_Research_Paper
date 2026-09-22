# AgroScore submission audit and handoff

## Outcome-mechanism redesign (rejection issue #2)

The generator now exposes four explicit outcome regimes through
`generate_simulation(..., outcome_regime=...)`: `additive_logit` (the original
reference mechanism), `threshold`, `interaction`, and `hybrid`.

Run `python regime_benchmark.py` to reproduce the ten-seed sensitivity study.
It writes `results2/regime_benchmark_raw.csv` and
`results2/regime_benchmark_summary.csv`. The manuscript now treats the old
seven-model result as a reference-regime case study and explicitly rejects a
universal “linear beats nonlinear” conclusion.

## Current verdict

The correct 28-feature reproduction code has now been identified. It matches the manuscript's seed, generator, outcome equation, ablations, learning curve and stress-test design. The previously supplied 77-feature v3 application files are a different system and should not be included as the paper's reproduction implementation.

The manuscript is close to submission-ready, subject to a clean full run with all pinned dependencies, a rebuilt PDF, an archival release and the target venue's formatting checks.

## Reproduction audit

| Item | Manuscript | Reproduction code | Status |
|---|---|---|---|
| Predictor count | 28 | 28 | matched |
| Main seed | 20260916 | 20260916 | matched |
| Compound-feature outcome coefficient | 0.85 | 0.85 | matched |
| Late-period shock | 0.35 for months 48--59 | same | matched |
| Ten-seed ablation | seeds 20260916--20260925 | same | rerun and matched |
| Learning curve | up to 32,000 records | same | rerun and matched |
| Random/OOT calibration | reported intercepts, slopes and ECE | same | rerun and matched |
| Temporal/regional/threshold tests | reported tables | same | rerun and matched |
| Seven-model benchmark | includes XGBoost 3.4.1 | code present | requires a clean environment with XGBoost installed |

The audit fixed missing calibration helper functions, missing ECE result columns, hard-coded output paths and a contradictory legacy execution path. It also corrected the cross-cohort distance calculation to fit one scaler on the reference cohort and transform both cohorts in the same coordinate system. The corrected DCR ratio is 0.9956, and the manuscript has been updated accordingly.

## Corrections already made in `agroscore.tex`

- Changed the title to distinguish the paper from a closely related 2026 preprint.
- Added and discussed the related outcome-conformant synthesis paper.
- Replaced the incorrect classifier two-sample-test citation.
- Replaced the weak SMOTE/leakage citation with a direct leakage reference.
- Updated the satellite-credit article's final publication details.
- Corrected the benchmark table's Brier-score emphasis.
- Corrected the model-difference figure caption; the stack interval crosses zero.
- Reframed the detector as model-specific evidence, not proof of equal distributions.
- Reframed DCR as a cross-cohort near-duplicate diagnostic, not a privacy or memorisation guarantee.
- Reframed marginal-sign, PCA and t-SNE checks to avoid causal, support and intrinsic-dimensionality claims.
- Qualified all model claims to the seven evaluated fixed configurations.
- Marked the learning curve as single-seed and descriptive.
- Removed the unsupported causal explanation for the out-of-time AUC increase.
- Added multiplicity, generator-parameter, metric-choice and single-seed limitations.
- Removed the inappropriate reviewer acknowledgment and preprint running header.
- Rewrote code/data availability so it no longer falsely says that the supplied files reproduce the results.
- Added a requirement for an immutable archive URL, version and licence.

## Files still required in Overleaf

The corrected source still requires the following items in the final Overleaf/repository package:

- the original `agroscore.bib`;
- all 13 figure PDFs under `figures/`;
- the corrected manuscript experiment and figure-generation code in this package;
- the generated cohort, predictions and machine-readable table results produced by `run_all.py`.

Also upload `agroscore_additions.bib`, because the revised source uses:

```latex
\bibliography{agroscore,agroscore_additions}
```

## Figure compliance

The submitted PDF contained Type 3 fonts in figures. Regenerate every Matplotlib PDF with TrueType embedding before recompiling:

```python
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
```

Then run IEEE PDF eXpress or the venue's PDF checker. Merely adding these settings to the LaTeX source does not repair already-exported figure PDFs.

## Citation and similarity checks

The newly added references were checked against their publisher/preprint metadata. Citation keys and LaTeX cross-references in the revised source are internally consistent.

This audit is not a substitute for iThenticate or Turnitin. A web/phrase review can identify obvious overlap, but only the venue's similarity system can report the submission's official similarity score. Review every highlighted match in context; bibliography entries, standard method names and properly cited technical phrases are not automatically plagiarism.

## Final manual checklist

1. Install `requirements.txt` and run `python run_all.py` from a clean environment. This begins with five deterministic unit tests.
2. Verify every number in the abstract, tables and captions against machine-readable output.
3. Upload the original bibliography and all figures, plus `agroscore_additions.bib`.
4. Regenerate figures with embedded TrueType fonts.
5. Insert the real archive DOI/URL, immutable version and licence.
6. Select the exact IEEE journal or conference and apply its page limit, author, biography, supplementary-material and double-blind rules.
7. If review is double-blind, remove author identity, affiliation, email and identifying repository metadata.
8. Run the compiled PDF through IEEE PDF eXpress and the venue's official similarity checker.
9. Clean any public repository descriptions that still claim real-data blending, novelty, publication readiness or results not supported by the submitted artifact.
10. Confirm the venue's current policy on disclosure of AI-assisted writing. No AI disclosure text was added in this revision at the author's request.

## Build status

The generator, ten-seed ablation, learning curve, calibration, temporal, regional and threshold analyses were rerun successfully and agree with the manuscript. The detector, KS, directional and PCA summaries also agree. The DCR preprocessing correction changed the ratio from 1.0000 to 0.9956 and is reflected in the source.

The seven-model benchmark could not be rerun in this workspace because the pinned XGBoost wheel was unavailable to the restricted package installer; its implementation is present and the official XGBoost documentation lists version 3.4.1. A full PDF rebuild also requires `IEEEtran.cls` and the 13 generated figure PDFs. Static checks found no missing or duplicated LaTeX labels/references. Compile the complete project in Overleaf after running the full pipeline.
