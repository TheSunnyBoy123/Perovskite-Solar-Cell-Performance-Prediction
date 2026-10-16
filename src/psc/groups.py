"""Feature provenance / interpretive group assignment (used for SHAP aggregation)."""
from .data import LIT32, MAT49

MAT = MAT49 | {"HTL_hole_barrier", "ETL_electron_barrier"}
COMP_PREFIX = ("a_", "b_", "halide_", "num_")
COMP_EXTRA = {"c_coeff_sum", "total_ion_count", "is_inorganic", "is_leadfree", "is_3D",
              "is_2D", "is_2D3D_mixture", "a_organic_fraction"}
DEVICE = {"Cell_architecture", "Add_lay_front", "Encapsulation", "is_flexible_x",
          "is_semitransparent", "Perovskite_band_gap_measured", "Perovskite_pl_max_nm"}
GROUPS = ("literature", "material", "composition", "process", "device", "noise")


def base(f):
    return f[:-3] if f.endswith("_na") else f


def group(f):
    b = base(f)
    if b.startswith("noise_"):
        return "noise"
    if b in LIT32:
        return "literature"
    if b in MAT:
        return "material"
    if b.startswith(COMP_PREFIX) or b in COMP_EXTRA or b.startswith("Perovskite_composition_"):
        return "composition"
    if b in DEVICE:
        return "device"
    return "process"
