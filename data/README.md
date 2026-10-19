# Data

Place the following files here before running any script:

| File | Description | Source |
|------|-------------|--------|
| `dataset_json_enriched.csv` | 43,398 PDP device records × 156 columns (PDP export + material properties + LLM recipe flags). SHA-256 is recorded in each run's `meta.json`. | Jacobsson et al., *Nat. Energy* 2022 (PDP); material properties from Materials Project API; recipe flags from `lit_v4.json` |
| `lit_v4.json` | Per-DOI LLM-derived recipe flags (36 features used by M3). | Provided in this repository |
| `pdp_ref_metadata.csv` | Lead author and publication date per `Ref_ID`. | PDP export |

## Not included

Publisher PDFs, their Markdown conversions, and raw LLM JSON responses cannot be redistributed due to publisher copyright. The derived features (`lit_v4.json`) are provided so the modelling results can be reproduced without re-running extraction.
