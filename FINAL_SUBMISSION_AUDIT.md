# AgroScore final pre-submission audit

## Verdict

**Status: submit after mandatory release checks, not immediately.**

The manuscript is now internally coherent and substantially more defensible as a synthetic-benchmark paper. The generator, reported non-XGBoost analyses and statistical caveats were checked against executable code. No remaining issue found in this audit invalidates the central scoped conclusion. Submission should nevertheless wait until the complete seven-model pipeline is run in a clean pinned environment, the regenerated PDF passes IEEE checks, and the final artifact receives a permanent release identifier.

## Audit results

| Area | Status | Evidence |
|---|---|---|
| Generator specification | Pass | 28 features, seed 20260916, coefficient 0.85 and shock 0.35 match source and manuscript |
| Cohort properties | Pass | 8,000 records; prevalence 0.123125; early/late prevalence 0.113281/0.162500 |
| Ten-seed ablation | Pass | All paper means, SDs and sign counts reproduced |
| Learning curve | Pass | All 18 values through 32,000 records reproduced |
| Random/OOT logistic results | Pass | AUC, AUC-PR, KS, Gini, Brier, ECE, intercept and slope reproduced |
| OOT non-XGBoost benchmark | Pass | Stack, logistic, random forest, extra trees, histogram boosting and MLP reproduced |
| Rolling-origin/LOGO/threshold tables | Pass | Values reproduced, including newly emitted ECE columns |
| Generator diagnostics | Pass with correction | Detector, KS, signs and PCA reproduced; shared-scaler DCR is 0.9956 and manuscript was corrected |
| Stack bootstrap intervals | Pass | AUC, AUC-PR, KS, Brier and ECE intervals reproduced over 2,000 paired resamples |
| XGBoost rows | Pending clean run | Code and pinned dependency are present; wheel unavailable in this restricted environment |
| Python syntax and unit tests | Pass | All official scripts compile; five deterministic tests pass |
| LaTeX references/labels | Pass | 0 missing references, 0 duplicate labels, 0 missing citation keys |
| BibTeX database syntax | Pass | Both databases parse successfully with BibTeX |
| Full LaTeX/PDF build | Pending | Local environment lacks IEEEtran class/style and final generated figure set |
| Similarity/plagiarism certification | Pending external check | Distinctive-phrase web spot checks found no matching manuscript; official iThenticate/Turnitin result is still required |

## Corrections made during this final audit

- Added ECE to random and out-of-time benchmark CSV outputs.
- Added ECE to rolling-origin and leave-one-region-out outputs.
- Added ECE to the 2,000-resample bootstrap output.
- Verified the manuscript's stack ECE point estimate and interval.
- Removed the contradictory legacy four-model/class-weighted execution path from the core module.
- Removed warning suppression from the official analysis path.
- Replaced hard-coded output paths and stale upload-directory imports.
- Made the out-of-time ROC label read its AUC from generated results instead of hard-coding it.
- Fixed the official t-SNE settings to match the stated 1,500-record reproduction path.
- Renamed the DCR output from `privacy.json` to `near_duplicate_audit.json`.
- Added five deterministic generator and PDO-score unit tests.
- Added standard algorithm citations for random forest, gradient boosting and XGBoost.
- Clarified that the bootstrap fraction is descriptive, not a posterior probability or adjusted p-value.
- Corrected the claim that the complete OOT model ordering was preserved.

## Mandatory actions before uploading to IEEE

1. In a clean environment, run:

   ```bash
   python -m pip install -r requirements.txt
   python run_all.py
   ```

   Confirm that it finishes without warnings or exceptions and produces all CSV/JSON files and 13 figure PDFs.

2. Compare the clean-run benchmark rows with Tables V and VII, particularly XGBoost. Do not submit if any rounded value differs.

3. Upload the generated `figures/` directory to Overleaf, compile twice after BibTeX, and inspect every page for clipped tables, unreadable labels and misplaced floats.

4. Run the final PDF through IEEE PDF eXpress. Confirm embedded fonts and no Type 3 fonts.

5. Replace the availability placeholder with an immutable DOI/URL, version/tag, licence and exact archive contents.

6. Select the exact IEEE venue. The existing PDF is approximately 16 journal-format pages; it will not fit a normal 6--8 page conference limit without major reduction.

7. If the venue is double-blind, remove the author name, affiliation, email and identifying repository/archive metadata from the review version.

8. Run the exact submitted PDF through the venue's similarity checker. A web search cannot certify a plagiarism percentage.

9. Confirm the venue's current disclosure policy for AI-assisted writing. No disclosure text was added at the author's request.

## Residual scientific limitations reviewers may raise

- All conclusions are conditional on an author-designed synthetic outcome process.
- Generator parameters are stylised rather than empirically estimated.
- The outcome is partly circular because model inputs generate the label.
- Hyperparameters are fixed rather than nested-tuned.
- The learning curve uses one seed at each sample size.
- The detector is model-specific and is not a distributional equivalence test.
- DCR depends on the selected scaling and Euclidean metric.
- LOGO evaluation isolates geography but not time.
- No real-portfolio fidelity, external validation or protected-group fairness claim is possible.

These limitations are now disclosed in the manuscript. They restrict the contribution but do not make a synthetic benchmarking paper inherently unpublishable.

## Similarity and naming note

The exact revised title and several distinctive manuscript phrases returned no matching paper in web spot checks. A separate public project named **AgriScore** exists for alternative smallholder-farmer credit scoring. This is a naming/discoverability collision, not evidence of copied text, but reviewers may confuse the projects. Consider adding a repository subtitle such as “AgroScore Synthetic Benchmark” consistently across the archive and README.
