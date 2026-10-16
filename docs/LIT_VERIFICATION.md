# Verification of Roberts (2024) and Fukasawa (2024) claims against the full texts

The automated literature audit (`audit/lit_audit.md`) could not open the publisher pages. The repository holds the open-access PDFs (`../Lit/fukasawa.pdf`, `../Lit/roberts.pdf`); text was extracted with pypdf and searched. Results:

## Fukasawa, Asahi, Taniguchi, Energy Adv. 3, 812 (2024) — all manuscript claims confirmed
| Manuscript claim | In the paper |
|---|---|
| 42,459 PDP records (as of 31 March 2022) | confirmed ("42 459 records as of 31 March 2022") |
| 36,937 curated records; 248 variables | confirmed ("we obtained 36 937 records"; "248 columns related to the material…") |
| Train/test split 8:2; 25 % of training data for validation during hyper-parameter search | confirmed ("split into 8 : 2"; "25% of the training dataset was used for validation") — the article does **not** describe grouping by publication |
| R² = 0.7177 (conventional split) | Table 2: 1st split, zero complement = 0.7177 |
| 294 newly registered records (as of 24 Aug 2023), R² 0.45, MAE 3.20 % | confirmed; the mean model on the same set gives R² −0.33, MAE 5.40 % |
| 6,446 degenerate points / 20,695 vectors | confirmed |
Best model: random forest (GBDT and NN lower). The manuscript's wording "an 8:2 train–test split (the article does not describe grouping by publication)" is accurate; it does not claim the split is random.

## Roberts et al., Adv. Phys. Res. 3, 2400060 (2024)
* Target is short-circuit current density (Jsc): confirmed ("short-circuit current density Jsc as a target"; PCE derived from Jsc × Voc is discussed but Jsc chosen).
* "an 80/10/10 split" (train/validation/test) is stated in the journal text (§ on train/test sets); no grouping by DOI is described.
* Duplicates: 18,320 valid entries drop to 7,154 after duplicate removal; only 154 duplicates exist in the raw database, so most duplicates arise during their feature reduction; duplicate groups are each "sourced from a unique paper" (verified by tagging with DOI). Replacement pseudo-Jsc = average of mean and median of the group's distribution.
* The manuscript no longer says the PDP contains "substantial duplication"; it says repeated configurations arose during Roberts' feature reduction.

## Still unverified
* Hussain et al. (RSC Adv. 2023): its split procedure is not stated in the article (read on PMC by the audit agent); the manuscript says exactly that.
* Existence of overlapping 2026 arXiv work on LLM-driven perovskite pipelines (arXiv:2602.04914, 2602.13312) — not read; check before claiming novelty for LLM-derived PSC features. The manuscript does not claim novelty for the LLM extraction itself.
