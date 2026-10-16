# lx3 feasibility check (no new extraction run)

Regex scan of the 2,835 cached extraction JSONs for the information lx3 would target:

| Target | Papers with any match |
|---|---|
| antisolvent mentioned | 898 (32%) — identity already captured in lx2 (`lx_chem_`) |
| additive / dopant / passivation | 1068 (38%) — already in lx2 |
| precursor concentration | 1079 (38%) — already in lx2 |
| antisolvent timing / volume | 0 (0%) |
| multistage spin (two rpm values) | 8 (0.3%) |
| annealing duration | 10 (0.4%) |

Conclusion: what is recoverable from existing JSON is already in lx2. Timing, multistage spin and
annealing schedules are absent from the JSONs; recovering them needs a new LLM pass over full text.
Only 22 Markdown full texts exist locally and no local LLM weights are cached, so that pass is
blocked pending a decision on the text source and LLM (external LLM use was never approved).
The confirm partition was already scored once with lx2, so an lx3 evaluation on it would not be untouched.
