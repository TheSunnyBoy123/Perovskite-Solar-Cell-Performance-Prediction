# Coverage-mask audit: earlier literature block vs recipe flags

Reproduce with `python scripts/audit_coverage_masks.py` (read-only; writes `results/coverage_mask_audit.json` and `.csv`, which lists the differing DOIs).

| mask | definition | records | publications |
|---|---|---:|---:|
| A: earlier 44-predictor block | publication has at least one non-missing of its 32 base columns (`results/feature_audit.json`) | 14,896 | 2,699 |
| B: recipe flags (`lx_covered`) | normalised DOI is among the 2,835 extraction records (`data/lit_v4.json`, `data/lit_v2.csv`) | 15,027 | 2,731 |

* Mask A is a strict subset of mask B (A minus B = 0 publications). B minus A = **32 publications / 131 records** (listed in `results/coverage_mask_audit.csv`).
* **Established cause.** All 32 have an extraction record but with **zero extracted materials** (`n_materials = 0`, no tokens, no numeric entries). The earlier block is missing for them in all 32 base columns, so mask A treats them as uncovered; the recipe-flag table keeps a row for them with all 36 flags equal to 0 (not missing), so mask B treats them as covered. 33 of the 2,835 extraction records have zero materials; 32 of these match the filtered dataset.
* **Consequence.** For these 131 records (0.9% of the 15,027 "covered" records) the recipe flags are 0 because the extraction was empty, not because the recipes were absent. The "covered" population of the V4 evaluation therefore means "an extraction record exists", which includes a small number of empty extractions. No result was recomputed for this audit.
* **Not recoverable here.** Why the 32 extractions are empty (model failure, unreadable PDF, review article, ...) cannot be determined because the raw JSON/PDF corpus is not part of the repository, and the original block-building script was not re-run.
