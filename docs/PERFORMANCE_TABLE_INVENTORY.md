# Inventory of performance tables and figures and their M1/M2/M3 coverage

Tier names: M1 (= t1) PDP features; M2 (= t2) + material properties; M3 (= t3) = M2 + the 36 V4 recipe flags. In result files the
final M3 is the tier key `t3f`; the tier key `t3` in `results/ablation_v1`, `strict_validation_v1`, `baselines_v1`, `model_dev_v1` is the
superseded 44-predictor literature block and is never shown as M3.

Status after this revision: **Done** = M3 arm run on the stored splits and integrated; **V4** = M3 already present (common split);
**M1/M2 only (labelled)** = M3 not run, scope stated in the caption and text; **n/a** = not a model-comparison item.

| Item | Purpose / protocol | M1 | M2 | M3 | Authoritative source | Status |
|---|---|---|---|---|---|---|
| Table 5 `tab:three`, Table 6 `tab:paired3`, Fig. 3 | Common reserved-partition comparison (matched default CatBoost, 5 seeds) | yes | yes | yes | `results/v4/final` | V4 (unchanged) |
| Table 7 `tab:v4`, Fig. 4 | Configuration dependence of M2->M3 (tuned / default / controls) | -- | yes | yes | `results/v4/final` | V4 (unchanged; M1 not tuned) |
| Table 8 `tab:tiers`, Fig. 5 | Five repeated DOI-grouped holdouts, pooled / p-i-n / n-i-p | yes | yes | **added** | `ablation_v1`, `t3final_matched_v1` | Done |
| `tab:delta` | Paired per-split tier differences (DOI-grouped) | yes | yes | **added (M2->M3)** | `ablation_v1/paired_bootstrap`, `t3final_summary/matched_pairs.csv` | Done |
| Table 9 `tab:baselines` | Reference models, pooled, DOI-grouped | ridge, CatBoost, XGB, RF, MLP, GPR | ridge, CatBoost | ridge, CatBoost **added** | `baselines_v1`, `t3final_matched_v1` | XGB/RF/MLP/GPR: M1 only (labelled); M3 suite planned in `configs/t3final_baselines.json`, **not run** pending decision |
| Table 10 `tab:dev`, SI S6 | Tuning, bagging, XGB, ensemble (model development) | yes | yes | no | `model_dev_v1` | M1/M2 only (labelled); M3 suite in `configs/t3final_model_dev.json`, **not run** pending decision |
| Fig. 6 `fig:dist_trio` | Predicted vs experimental distribution, CatBoost / MLP / GPR, split 1 | yes (all three models) | -- | -- | `ablation_v1`, `baselines_v1` | M1 only (labelled; MLP/GPR not run on M3) |
| Fig. 7 `fig:parity` | Parity and residuals, stacked five splits | -- | earlier | **now M3** | `t3final_matched_v1/preds` | Done (M2 statistics quoted alongside) |
| Table 11 `tab:split`, Fig. 8 | Record-level vs DOI-grouped (CatBoost, ridge) | yes | yes | **added** | `ablation_v1`, `random_convergence_cap20000`, `t3final_matched_v1` | Done (M3 record-level pooled uses the 20,000-iteration cap) |
| Table 12 `tab:strict` | Record-level / DOI / lead-author / forward (CatBoost, ridge, mean) | yes | yes | **added** | `strict_validation_v1`, `t3final_matched_v1` | Done |
| Table 13 `tab:temporal` | Per-window forward-in-time R2 | yes | yes | **added** | same | Done |
| Table 14 `tab:t3protocols`, Fig. 9 (new) | Paired M3-M2 across all protocols | -- | yes | yes | `t3final_summary/matched_summary.csv` | New |
| Fig. 10 (new) | SHAP group shares of the final M3 model | -- | -- | yes | `results/shap/t3final_*` | New (earlier SHAP of the 44-predictor model stays in SI Figs. S1-S2, labelled) |
| SI S2 | Per-split DOI-grouped CatBoost | yes | yes | earlier block (labelled) | `ablation_v1` | Final M3: SI S14-S17 |
| SI S5, S6 | Earlier 44-predictor block experiments | -- | -- | earlier block (labelled) | `ablation_v1`, `baselines_v1`, ... | Historical, unchanged |
| SI S7, S9-S12 | V4 per-seed deltas, illumination, publication-weighted, unpenalised | yes | yes | yes | `results/v4/final`, `results/revision` | V4 / sensitivity (unchanged) |
| SI S14-S17 (new) | M3 run integrity; per-split pairs; per-architecture summary; SHAP of M3 | -- | yes | yes | `t3final_*` | New |
| Tables 1-4, 15; Fig. 1-2 | Dataset, tiers, configuration, protocol, literature comparison, workflow | n/a | n/a | n/a | -- | No tier rows needed (Table 15 gets the M3 numbers of this study) |

Reproduce: `python scripts/run_t3_final.py --config configs/t3final_matched.json` (108 fits, about 1.5 h on a laptop),
`python scripts/t3_protocol_summary.py`, `python scripts/make_t3_tables.py`, `python scripts/shap_t3final.py`,
`python psc_manuscript/make_figures.py`, `python scripts/make_si.py`, `python scripts/verify_t3_numbers.py`.
