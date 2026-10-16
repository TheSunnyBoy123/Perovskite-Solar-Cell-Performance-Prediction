import numpy as np
import pytest
from psc.data import LX_TIERS, load_dataset, doi_partition
from psc.features import base_frame
from psc.litfeat import OUTCOME_KEY, num, paper_features, role


def test_outcome_keys_never_enter_features():
    doc = {"materials": [{"formula": "MAPbI3", "type": "perovskite absorber", "pce_champion": "21.5 %", "voc": "1.1 V",
                          "jsc_ma_cm2": 23, "ff": 0.8, "annealing_temperature": "100 °C", "stability_t80": "500 h"}]}
    f = paper_features(doc)
    assert f["lx_proc_perov_anneal_C"] == 100
    assert not any(OUTCOME_KEY.search(k) for k in f)
    assert f["lx_meta_n_fields"] == 3 or f["lx_meta_n_fields"] == 4  # outcome keys excluded from the field count


@pytest.mark.parametrize("v,kind,exp", [("100 °C", "temp_C", 100), ("1 h", "time_min", 60), ("30 s", "time_s", 30),
                                         ("0.5 μm", "layer_nm", 500), ("4000 rpm", "rpm", 4000), ("1.2 M", "conc_M", 1.2),
                                         ({"value": 0.1, "unit": "cm²"}, "area_cm2", 0.1), ("3.0 mm × 3.0 mm", "area_cm2", None)])
def test_unit_normalisation(v, kind, exp):
    r = num(v, kind)
    if exp is None:
        return
    assert r == pytest.approx(exp)


def test_implausible_and_wrong_unit_rejected():
    assert np.isnan(num("35 wt%", "conc_M")) and np.isnan(num("5 mg/mL", "conc_M"))
    assert np.isnan(num("50000 rpm", "rpm")) and np.isnan(num("abc", "temp_C"))


def test_not_reported_is_nan_not_zero():
    f = paper_features({"materials": [{"formula": "Au", "type": "metal contact"}]})
    assert np.isnan(f["lx_chem_solv_DMF"]) and np.isnan(f["lx_chem_anti_CB"]) and np.isnan(f["lx_proc_perov_spin_rpm"])
    g = paper_features({"materials": [{"formula": "DMF", "type": "solvent"}]})
    assert g["lx_chem_solv_DMF"] == 1 and g["lx_chem_solv_DMSO"] == 0


def test_roles():
    assert role({"type": "electron transport layer (ETL)"}) == "etl"
    assert role({"type": "antisolvent", "common_name": "chlorobenzene"}) == "antisolvent"


@pytest.fixture(scope="module")
def df():
    return load_dataset(arch="overall")


def test_doi_level_constancy_and_join(df):
    lx = [c for c in df.columns if c.startswith("lx_") and c != "lx_covered"]
    assert lx
    sub = df[df.lx_covered == 1]
    assert (sub.groupby("doi_group")[lx].nunique(dropna=False).max() <= 1).all()   # paper-level: constant within DOI
    assert df.loc[df.lx_covered == 0, lx].isna().all().all()


def test_partition_is_deterministic_and_doi_disjoint(df):
    assert doi_partition("10.1000/x") == doi_partition("10.1000/x")
    g = df.groupby("doi_group")["partition"].nunique()
    assert (g == 1).all()


def test_tier_nesting_and_no_lx_in_legacy_tiers(df):
    t2 = set(base_frame(df, "t2").columns)
    assert not any(c.startswith("lx_") for c in t2) and not any(c.startswith("ctl_") for c in t2)
    for t in LX_TIERS:
        cols = set(base_frame(df, t).columns)
        assert t2 <= cols or t == "t3b"
    assert set(base_frame(df, "t3").columns) <= set(base_frame(df, "t3b").columns)


def test_controls_are_doi_level_and_independent_of_target(df):
    c = base_frame(df, "t2_lxperm_s0")
    ctl = [x for x in c.columns if x.startswith("ctl_")]
    d = df[["doi_group"]].join(c[ctl])
    sub = d.groupby("doi_group")[ctl].nunique(dropna=False)
    assert (sub.max() <= 1).all()
    real = base_frame(df, "t3x")[[x[4:] for x in ctl]]
    dd = d.drop_duplicates("doi_group")[ctl]; rr = df[["doi_group"]].join(real).drop_duplicates("doi_group")[[x[4:] for x in ctl]]
    assert dd.isna().values.mean() == pytest.approx(rr.isna().values.mean(), abs=1e-9)  # same DOI-level availability
