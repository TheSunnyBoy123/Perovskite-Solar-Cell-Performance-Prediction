# Suggested manuscript paragraph (preliminary; not human-adjudicated)

We carried out a preliminary, model-assisted audit of the 36 recipe flags used in M3 against the
full text of the publications that could be checked. Of the 40 publications we had planned to
audit, only 17 had both a full-text source and a cached extraction record, so the audit is a
census of that available set rather than a random sample. For each publication and flag we
recorded whether the paper's own devices used the recipe element, using verbatim evidence drawn
from the source text. The resulting 612 paper-flag decisions yielded 603 resolved decisions (9
unresolved) comprising 84 source-supported positives and 519 negatives. Against these resolved
decisions, the flags reproduced the source with 90.0% agreement, a micro precision of 0.74 (95%
exact CI 0.60–0.85) and a micro recall of 0.44 (0.33–0.55). Only seven flags had enough source
positives and predicted positives for an estimate to be meaningful, and the confidence intervals
for them are wide. Most discrepancies were false negatives in which device use was absent from the
extraction; the principal deterministic-conversion errors were regex over-breadth (notably
methylammonium matched by the PEA-family pattern) and missed aliases. Because the sample is not
random and the source labels were produced by a model reading the text rather than by human
annotators, these figures describe this set of 17 publications only. They are not estimates of
performance across the corpus, and a human adjudication of the flagged decisions is required before
they are used as validation evidence.
