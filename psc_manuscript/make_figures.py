"""Master entry point: regenerates every data figure used by ml.tex into psc_manuscript/figures/.

    python make_figures.py              # all figures
    python make_figures.py tiers split  # selected figures (tiers, split, diag, dist, shapgroups, shap, v4)

Each figure has its own editable script in figure_scripts/ (plotting logic, data source and output name are documented at
the top of each file); shared fonts, colours and save settings live in figure_scripts/style.py. Figures read the
authoritative results under ../results/ and hard-code no result values.
device_structure.png and ml_pipeline.png are not data figures (TikZ/drawing sources under tikz/ and scripts/build_workflow_figure.py).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "figure_scripts"))
import fig_feature_tiers, fig_split_effect, fig_oof_diagnostics, fig_pce_dist_trio, fig_shap_groups, shap, fig_v4_recipe

FIGS = {"tiers": fig_feature_tiers, "split": fig_split_effect, "diag": fig_oof_diagnostics, "dist": fig_pce_dist_trio,
        "shapgroups": fig_shap_groups, "shap": shap, "v4": fig_v4_recipe}

if __name__ == "__main__":
    for k in (sys.argv[1:] or FIGS):
        FIGS[k].main()
