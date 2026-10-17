# Manuscript version history (psc_manuscript/)

Versions are tracked by annotated Git tags `ms-vX.Y` on the commit that produced the PDFs, and by frozen copies of the compiled
PDFs in `versions/` (`ml_vX.Y.pdf`, `si_vX.Y.pdf`). The working files `ml.tex`, `si.tex`, `ml.pdf`, `si.pdf` always hold the latest
version. Tags can be listed with `git tag -n1 'ms-v*'`; a version can be recovered with `git show ms-vX.Y:psc_manuscript/ml.tex`.

| Version | Tag / commit | Content |
|---|---|---|
| v1.0 | `ms-v1.0` (fa08745) | Initial project snapshot. |
| v2.0 | `ms-v2.0` (d550f11) | t1/t2/t3 unified around the V4 recipe features. |
| v2.1 | `ms-v2.1` (024ab45) | Coverage-mask audit and retrieval-provenance statement. |
| v3.0 | `ms-v3.0` (62d87bd) | Final revision: new title and narrative, corrected provenance/target/leakage/count text, material-descriptor implementation, related work, SI S8-S13. |
| v3.1 | `ms-v3.1` | Notation: tier symbols t_1, t_2, t_3 replaced by $\mathcal{M}_1$, $\mathcal{M}_2$, $\mathcal{M}_3$ (949089b). |
| v4.0 | `ms-v4.0` | Final M3 added to the repeated DOI-grouped, record-level, lead-author and forward-in-time evaluations, parity diagnostics and SHAP; new Section 3.6 (Table 14, Fig. 9), Fig. 10, SI S14-S17; claims revised to report the protocol dependence of the recipe-flag effect. Reference-model and model-development suites for M3 not run. |

PDF copies exist in `versions/` for v3.0 and later; earlier versions are recoverable from the tags.
