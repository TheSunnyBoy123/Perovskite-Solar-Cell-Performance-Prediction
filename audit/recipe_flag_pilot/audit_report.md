# Recipe-flag source audit: 17-paper pilot (M3, 36 LLM-derived recipe flags)

Status: **model-assisted pilot, not human-adjudicated.** All source-level labels were written by
Claude from cached Markdown full text. No label here has been reviewed by a human. Items that need
human confirmation are listed in `human_review_required.csv`.

## 1. Scope and what was checked

- **Flag definitions.** The 36 flags are `RECIPE` in `src/psc/litfeat4.py`. `paper_recipe()` marks a
  flag 1 if any non-outcome string in a paper's extraction records matches the regex, with role
  scoping for `htl_*`/`etl_*` flags. The flags are deterministic and are not direct LLM outputs.
- **Pipeline check.** Recomputing the flags from `opencode_sandbox/combined_extracted_features.json`
  with the production function reproduces the stored `data/lit_v4.json` flags exactly for all 2,782
  shared DOIs (0 papers differ in any flag; `frame_check.json`). The combined JSON and `lit_v4.json`
  each contain 53 DOIs that the other lacks.
- **Full-text availability.** `sandbox/markdown` holds 22 full texts and `sandbox/pdfs` holds 21 PDFs
  (one Markdown file has no PDF). The `Lit/` PDFs are not keyed to extraction DOIs and were not used.
  The 80-publication worksheet DOIs have no full text in the sandbox.
- **Auditable frame.** 17 papers have both a Markdown full text and an extraction record. Five full
  texts (10.1016/j.electacta.2017.12.108, 10.1021/jp5126624, 10.1117/12.2506443,
  10.1149/08701.0199ecst, 10.1246/cl.180616) have no extraction record and are excluded
  (`sample_manifest.csv`).

## 2. Sampling: a census, not a sample

The target was 40 papers. The auditable frame contains only 17, so the target was not reachable
without a full-text acquisition campaign, which the task excludes. All 17 were audited. No random
draw was made, so **no seed applies**. Selection was determined by source availability, not by flag
values or audit outcomes.

Because this is a census of a source-availability frame, it is not representative of the 2,835-paper
corpus. Results describe these 17 papers only and must not be presented as corpus-wide performance.

## 3. Audit procedure

For each of the 612 paper-flag pairs (17 × 36):

1. The source label is one of:
   - **P (supported positive):** the paper's own devices or fabrication use the element. Borderline
     cases are flagged, for example a reagent used to synthesise a precursor.
   - **N (supported negative):** the term is absent, or appears only in a non-qualifying context.
     Each negative has a reason code (`REASONS` in `scripts/audit_common.py`): literature,
     reference list, artifact, solvent, electrode, test condition, and so on.
   - **U (unresolved):** the text cannot settle the decision without adjudication.
2. Every P decision is anchored to a phrase in the source. The evidence string is extracted verbatim
   by the build script, not typed by hand. The build fails if an anchor is absent.
3. Every flag with any regex hit in the full text must have an explicit decision. The build refuses
   to run otherwise, so every visible hit was reviewed.

The operational definition of a positive is that the element is used in the devices the paper
reports. Mention in the introduction, the references, or a literature comparison counts as negative.
This interpretation is a choice made for the audit and should be confirmed by a reviewer.

## 4. Results

| Quantity | Value |
|---|---|
| Papers audited | 17 of 40 target (census) |
| Paper-flag decisions | 612 |
| Resolved (P or N) | 603 (98.5%) |
| Unresolved (U) | 9 (1.5%); in 8 of 17 papers |
| Papers with every decision resolved | 9 of 17 |
| Source-supported positives / negatives | 84 / 519 |
| Pooled TP / FP / TN / FN | 37 / 13 / 506 / 47 |
| Agreement on resolved decisions | 90.0% |
| Micro precision (exact 95% CI) | 0.74 (0.60–0.85) |
| Micro recall (exact 95% CI) | 0.44 (0.33–0.55) |
| Macro precision (21 flags with defined precision) | 0.77 |
| Macro recall (24 flags with defined recall) | 0.48 |
| Flags with any source-supported positive | 24 of 36 |
| Flags with adequate support (≥3 source positives and ≥3 predicted positives) | 7 |

The 12 flags with no source-supported positive in this set are `scn`, `snf2`, `ava`, `water_add`,
`dio`, `htl_f4tcnq`, `htl_cu`, `htl_borate`, `hot_cast`, `solvent_anneal`, `flexible` and `tandem`.
Recall and precision are undefined for them and are reported as NA.

### Per-flag estimates for the seven adequately supported flags

| Flag | TP | FP | FN | TN | Precision (95% CI) | Recall (95% CI) |
|---|---|---|---|---|---|---|
| cl_source | 4 | 0 | 0 | 13 | 1.00 (0.40–1.00) | 1.00 (0.40–1.00) |
| cs | 4 | 0 | 2 | 11 | 1.00 (0.40–1.00) | 0.67 (0.22–0.96) |
| lewis_base | 3 | 0 | 4 | 10 | 1.00 (0.29–1.00) | 0.43 (0.10–0.82) |
| carbon | 5 | 0 | 3 | 9 | 1.00 (0.48–1.00) | 0.63 (0.25–0.92) |
| htl_tbp | 3 | 0 | 3 | 11 | 1.00 (0.29–1.00) | 0.50 (0.12–0.88) |
| htl_litfsi | 3 | 0 | 4 | 10 | 1.00 (0.29–1.00) | 0.43 (0.10–0.82) |
| interface_pass | 3 | 0 | 0 | 13 | 1.00 (0.29–1.00) | 1.00 (0.29–1.00) |

Intervals are wide because the counts are small. For the remaining flags, `audit_summary.csv`
gives the counts, denominators and unresolved numbers, and `figures/fig1_flag_precision_recall`
marks them as low-support. Those estimates should not be interpreted.

## 5. Where the errors arise

Of the 60 discrepant decisions (47 false negatives, 13 false positives), the cause is classified
as follows:

- **Extraction (44).** 39 false negatives are device use that the JSON does not record at all
  (omission). The remaining extraction errors are records that the JSON attaches to a flag in a
  non-device context, such as literature, a reference, a test condition, or an etch step.
- **Deterministic conversion (16).** These arise where the JSON contains the material but the
  regex or role mapping does not capture it:
  - Missed aliases: `PC71BM` (regex `pcbm` misses it, two papers), `DMSO` (regex `sulfoxide` only),
    the bare `PEA` abbreviation inside `(PEA)2(MA)…`, `Cs0.05` formulae (regex `cs` requires CsI,
    CsBr, CsPb or cesium).
  - Role scoping: a Sn-doped TiO2 photo-anode is mapped to role `contact` by `role()`, which is
    outside the `etl_dopant` scope.
  - Over-breadth (the regex matches a non-target string): `ammonium` matches methylammonium,
    giving most of the `pea_family` false positives (ten predicted positives in total, of which one
    is correct); `pristine` marks as-prepared films as `undoped`; `doped` matches the FTO substrate
    and PEDOT:PSS; `quench` matches photoluminescence quenching.

The most frequent false-negative pattern is device use absent from the JSON, which affects
process-environment and HTL-additive flags (`n2_box`, `htl_tbp`, `htl_litfsi`, `two_step`).

The main false-positive pattern is the `pea_family` regex. Methylammonium is not a PEA-family cation,
so the flag fires on most methylammonium papers.

Flag definitions were not changed, as required. The following minimal fixes are proposed for a
separate decision, with re-running M3 needed for any of them:
- `pea_family`: exclude `methyl`-prefixed ammonium (for example `(?<!methyl)ammonium`) and add the
  bare `PEA` formula token.
- `lewis_base`: add `dmso`; `carbon`: add `pc71bm`; `cs`: add `cs0`.
- `etl_dopant`: include the `ctc` role for photo-anodes, or map photo-anodes to `etl` in `role()`.
- `hx_acid`: add `hydrochloric` and `hydroiodic`, and decide whether precursor-synthesis reagents count.

## 6. Limitations

1. **Census, not random.** Metrics describe these 17 papers only. The 2,835-paper corpus was not
   sampled.
2. **Model-assisted labels.** The source labels are Claude's reading of Markdown full text, not
   human annotation. The text is the cached Markdown, not a page-by-page re-read of the PDFs.
3. **Small denominators.** Only 7 flags have adequate support, and the intervals for them are wide.
4. **Definition-dependent labels.** Borderline cases include HI/HBr/HCl used as precursor reagents,
   PEDOT:PSS as a polymer layer, gas quenching as a quench method, and LiTFSI treatment of a TiO2
   layer. These are flagged in `human_review_required.csv`. Their labels move the metrics.
5. **Alias attribution is approximate.** The "alias present in JSON" check is a substring test, so
   the conversion-versus-extraction split is an approximation. The specific causes listed in
   section 5 were verified against the JSON text.
6. **Nine decisions are unresolved** and are excluded from all metrics, not counted as correct.

## 7. Human review

`human_review_required.csv` lists 73 decisions, ordered by priority:
- priority 1: the 9 unresolved decisions;
- priority 2: discrepant decisions and borderline decisions;
- priority 3: borderline negatives.

Each row carries the verbatim evidence, the reason code and the JSON match. Empty columns
`human_decision`, `human_reviewer` and `human_notes` are for the reviewer. No human has filled them.

## 8. Files

- `sample_manifest.csv`: 17 included papers and 5 excluded, with reasons.
- `recipe_flag_audit.csv`: 612 rows, one per paper-flag decision, with evidence and provenance.
- `audit_summary.csv`: per-flag counts, denominators, intervals and support status.
- `metrics_overall.json`: pooled and macro summaries, and discrepancy counts.
- `frame_check.json`: reproduction of `lit_v4.json` flags.
- `human_review_required.csv`: adjudication queue.
- `figures/`: `fig1_flag_precision_recall`, `fig2_confusion_heatmap`, `fig3_coverage_errors`,
  `fig4_evidence_panel` (each as PDF and PNG).
- `scripts/`: `audit_common.py`, `source_decisions.py` (the source judgements),
  `build_audit.py`, `compute_metrics.py`, `plot_audit.py`.
- `tests/test_recipe_flag_audit.py` (in `ml_paper/tests/`): 8 lightweight tests, all passing.

Reproduce with `python3 scripts/build_audit.py && python3 scripts/compute_metrics.py && python3 scripts/plot_audit.py`
from this directory.
