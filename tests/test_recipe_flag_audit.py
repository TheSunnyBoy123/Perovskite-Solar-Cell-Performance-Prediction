"""Lightweight checks for the recipe-flag source audit (audit/recipe_flag_pilot/scripts)."""
import csv
import math
import sys
from pathlib import Path

import pytest

PILOT = Path(__file__).resolve().parents[1] / "audit" / "recipe_flag_pilot"
sys.path.insert(0, str(PILOT / "scripts"))

from audit_common import FLAGS, JSON_ALIASES, flatten, json_alias_present  # noqa: E402
from compute_metrics import clopper_pearson, flag_row  # noqa: E402
from source_decisions import DECISIONS  # noqa: E402


def test_flag_list_is_the_36_m3_flags():
    assert len(FLAGS) == 36 and len(set(FLAGS)) == 36


def test_flatten_collapses_whitespace():
    assert flatten("a\n\n  b\tc") == "a b c"


def test_alias_matching_is_word_bounded_for_short_terms():
    assert not json_alias_present("htl_litfsi", "adahcl")   # 'hcl' inside 'adahcl' is not a match
    assert json_alias_present("cl_source", '"formula": "pbcl2"')
    assert set(JSON_ALIASES) <= set(FLAGS)


def test_decisions_reference_known_flags_only():
    for doi, flags in DECISIONS.items():
        assert doi.startswith("10."), doi
        assert set(flags) <= set(FLAGS), (doi, set(flags) - set(FLAGS))


def test_clopper_pearson_bounds():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35
    lo, hi = clopper_pearson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.75
    lo, hi = clopper_pearson(5, 0)
    assert math.isnan(lo) and math.isnan(hi)


def _row(flag, cmp, borderline="0"):
    return {"flag": flag, "comparison": cmp, "borderline": borderline, "needs_human_review": "0"}


def test_zero_denominators_are_not_reported_as_numbers():
    rows = [_row("x", "TN"), _row("x", "TN"), _row("x", "UNRESOLVED")]
    r = flag_row("x", rows)
    assert math.isnan(r["precision"]) and math.isnan(r["recall"])
    assert r["unresolved"] == 1 and r["resolved"] == 2


def test_precision_and_recall_counts():
    rows = [_row("y", "TP"), _row("y", "TP"), _row("y", "FP"), _row("y", "FN"), _row("y", "TN")]
    r = flag_row("y", rows)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["F1"] == pytest.approx(4 / 6)


@pytest.mark.skipif(not (PILOT / "recipe_flag_audit.csv").exists(), reason="audit tables not built")
def test_saved_audit_is_complete_and_consistent():
    rows = list(csv.DictReader(open(PILOT / "recipe_flag_audit.csv", encoding="utf8")))
    assert len(rows) == 17 * 36
    for r in rows:
        if r["source_label"] == "U":
            assert r["comparison"] == "UNRESOLVED"
        if r["source_label"] == "P":
            assert r["evidence_verbatim"] and r["evidence_verbatim"] != "no regex hit in full text"
