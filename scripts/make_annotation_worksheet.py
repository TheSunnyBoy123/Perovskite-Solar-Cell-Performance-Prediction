#!/usr/bin/env python3
"""Human-validation worksheet for the 36 recipe flags (NOT a validation result: all reviewer columns are blank).

Samples 80 publications (20 per quartile of PDP publication year) from the filtered dataset that have a non-empty extraction
record; for each, lists every positive flag with the matching text span from the cached extraction JSON plus 5 randomly chosen
negative flags, with blank columns for a human to judge against the source paper. Output: docs/validation/ (local only; it
contains short spans of raw LLM output, which are not redistributed, so the file is git-ignored)."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset
from psc import v4
from psc.litfeat import OUTCOME_KEY, role
from psc.litfeat4 import ROLE_MAP, RECIPE, _RECIPE, RECIPE_SCOPE
GROUP = {g: set(c) for g, c in v4.REC_GROUPS.items()}
GNAME = {"P": "additive/precursor", "H": "HTL dopant", "E": "ETL treatment", "R": "process route", "C": "device context", "U": "undoped/interface"}
flag_group = {c: GNAME[g] for g, cs in v4.REC_GROUPS.items() for c in cs}
df = load_dataset(); P = v4.papers()
pubs = df[df.doi_group != "__NO_DOI__"].groupby("doi_group").agg(year=("pub_date", lambda s: s.min().year), n_records=("final_pce", "size")).reset_index()
pubs = pubs[pubs.doi_group.isin([d for d, p in P.items() if p["recipe"]["n_materials"] > 0])]
pubs["bin"] = pd.qcut(pubs.year.rank(method="first"), 4, labels=False)
rng = np.random.default_rng(2026)
sel = pd.concat([g.sample(20, random_state=int(rng.integers(1e9))) for _, g in pubs.groupby("bin")])
JD = ROOT.parent / "json"
def records(doc):
    out = []
    for m in doc.get("materials", []):
        if not isinstance(m, dict): continue
        r = ROLE_MAP.get(role(m), "oth"); parts = []
        for k, v in m.items():
            kl = str(k).lower()
            if OUTCOME_KEY.search(kl): continue
            parts.append(kl.replace("_", " "))
            if isinstance(v, (str, int, float)): parts.append(str(v))
            elif isinstance(v, list): parts += [str(x) for x in v if isinstance(x, (str, int, float))]
        out.append((r, " ".join(parts).lower()))
    return out
def find_json(doi):
    f = JD / (doi.replace("/", "_") + ".json")
    return f if f.exists() else None
rows = []
for _, p in sel.iterrows():
    f = find_json(p.doi_group)
    if f is None: continue
    recs = records(json.loads(f.read_text())); rc = P[p.doi_group]["recipe"]
    pos = [k for k in RECIPE if rc.get(k) == 1.0]; neg = [k for k in RECIPE if rc.get(k) == 0.0]
    chosen = [(k, 1) for k in pos] + [(k, 0) for k in rng.choice(neg, size=min(5, len(neg)), replace=False)]
    for k, val in chosen:
        scope = next((sc for pre, sc in RECIPE_SCOPE.items() if k.startswith(pre)), None)
        span, role_ = "", ""
        if val:
            for r, t in recs:
                if scope is not None and r not in scope: continue
                m = _RECIPE[k].search(t)
                if m:
                    span = t[max(0, m.start() - 70): m.end() + 70].replace("\n", " "); role_ = r; break
        rows.append(dict(doi=p.doi_group, pub_year=int(p.year), n_device_records=int(p.n_records), flag=f"rc_{k}", flag_group=flag_group[k],
                         flag_definition_regex=RECIPE[k], extracted_value=int(val), matched_record_role=role_, evidence_span=span,
                         reviewer_flag_correct_in_paper="", reviewer_flag_value_in_paper_0_1="", reviewer_evidence_ok="", reviewer_notes=""))
W = pd.DataFrame(rows)
out = ROOT / "docs/validation"; out.mkdir(exist_ok=True)
W.to_csv(out / "recipe_flag_annotation_worksheet.csv", index=False)
(out / "README.md").write_text("""# Recipe-flag annotation worksheet

`recipe_flag_annotation_worksheet.csv` (local, git-ignored: contains short spans of raw LLM output) is a blank review form, NOT a validation result.

* Sample: %d publications, 20 per quartile of the PDP publication year, drawn from filtered-dataset publications with a non-empty extraction record (rng seed 2026).
* Rows: every positive flag of the publication (with the matching span of the cached extraction JSON) plus up to 5 randomly chosen negative flags.
* A reviewer reads the source paper and fills `reviewer_flag_value_in_paper_0_1` (does the paper report that recipe element for the devices studied?), `reviewer_flag_correct_in_paper` (1 if it agrees with `extracted_value`), `reviewer_evidence_ok` and notes.
* Report precision on positive rows and the false-negative rate on the sampled negative rows only after at least two reviewers have completed it; until then no validation accuracy may be claimed.
""" % sel.shape[0])
print(len(W), "rows;", W.doi.nunique(), "publications;", int(W.extracted_value.sum()), "positive rows")
