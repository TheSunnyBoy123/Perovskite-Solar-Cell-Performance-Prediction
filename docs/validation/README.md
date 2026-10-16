# Recipe-flag annotation worksheet

`recipe_flag_annotation_worksheet.csv` (local, git-ignored: contains short spans of raw LLM output) is a blank review form, NOT a validation result.

* Sample: 80 publications, 20 per quartile of the PDP publication year, drawn from filtered-dataset publications with a non-empty extraction record (rng seed 2026).
* Rows: every positive flag of the publication (with the matching span of the cached extraction JSON) plus up to 5 randomly chosen negative flags.
* A reviewer reads the source paper and fills `reviewer_flag_value_in_paper_0_1` (does the paper report that recipe element for the devices studied?), `reviewer_flag_correct_in_paper` (1 if it agrees with `extracted_value`), `reviewer_evidence_ok` and notes.
* Report precision on positive rows and the false-negative rate on the sampled negative rows only after at least two reviewers have completed it; until then no validation accuracy may be claimed.
