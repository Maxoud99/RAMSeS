"""
Read-only reader over the explainability artifacts.

Assembles the page payload from the per-stage `nl_*.txt` and `ir_*.json` files
rather than parsing `nl_global_iter*.txt`. The global text is *derived* from the
same per-stage strings by `Explainability.llm.compose_global_narrative`, so
parsing it back would be a lossy round-trip through a format whose section
separators are `"=" * len(title)` — any wording change there would silently
break the reader with no test failure. Assembling instead also yields the
structured data the text cannot carry (decision, stage agreement, per-stage
outputs, caveats, confidence), which is what makes the page readable rather than
a 2,000-word wall. The global `.txt` stays available as a verbatim download.
"""

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Utils.pipeline_spec import dataset_label
from WebUI import paths
from WebUI.summarize import attribute_sentences, summarize

# ── The three-vocabulary map ────────────────────────────────────────────────
#
# Three different names exist for the same stage: the CLI --stages token, the
# IR/NL stage key, and the plot directory. This tuple is the single owner of
# that mapping; plots.py, markers.py and the frontend all join on `key`.
# Order matches Explainability.llm._GLOBAL_STAGE_ORDER.
STAGES: Tuple[Dict[str, Any], ...] = (
    {"key": "ga_selection", "title": "Ensemble selection (genetic algorithm)",
     "cli": "ga", "ir": "ir_ga_selection", "nl": "nl_ga_selection",
     "plot_group": "ga_selection", "order": 1},
    {"key": "ga_combination", "title": "Ensemble weighting (meta-learner)",
     "cli": "ga", "ir": "ir_ga_combination", "nl": "nl_ga_combination",
     "plot_group": "ga_combination", "order": 2},
    # One CLI token, two stages — the same split as ga_selection/ga_combination.
    # `thompson_ranking` explains mu^T mu, the criterion the detectors are
    # ordered by; `thompson_sampling` explains mu^T x, the expected reward that
    # drove per-window selection. Neither keeps the plain name.
    #
    # `plot_group` is deliberately `ts_ranking`, not `thompson_ranking`:
    # result.js matches lazy-gallery descriptors with `id.startsWith(plot_group)`,
    # so a group prefixed by "thompson" would let one card claim the other's
    # galleries. `regimes` names the plot subdirectory whose per-regime figures
    # pair with this stage's regime atoms; its presence is what makes a stage
    # regime-bearing, replacing a hardcoded stage-key check here and in server.py.
    {"key": "thompson_ranking", "title": "Thompson Sampling: ranking criterion",
     "cli": "thompson", "ir": "ir_thompson_ranking", "nl": "nl_thompson_ranking",
     "plot_group": "ts_ranking", "order": 3,
     "regimes": ["ranking_per_regime"]},
    {"key": "thompson_sampling", "title": "Thompson Sampling: selection dynamics",
     "cli": "thompson", "ir": "ir_thompson", "nl": "nl_thompson",
     "plot_group": "thompson", "order": 4,
     "regimes": ["reward_per_regime", "shap_per_regime"]},
    {"key": "monte_carlo", "title": "Robustness: Monte Carlo noise sweep",
     "cli": "montecarlo", "ir": "ir_monte_carlo", "nl": "nl_monte_carlo",
     "plot_group": "monte_carlo", "order": 5},
    {"key": "off_by_threshold", "title": "Sensitivity: off-by-threshold test",
     "cli": "offby", "ir": "ir_off_by", "nl": "nl_off_by",
     "plot_group": "off_by", "order": 6},
    {"key": "rank_aggregation_robust", "title": "Robustness consensus",
     "cli": None, "ir": "ir_rank_aggregation_robust", "nl": "nl_rank_aggregation_robust",
     "plot_group": "rank_aggregation_robust", "order": 7, "iterated": True},
    {"key": "rank_aggregation_final", "title": "Final consensus",
     "cli": None, "ir": "ir_rank_aggregation_final", "nl": "nl_rank_aggregation_final",
     "plot_group": "rank_aggregation_final", "order": 8, "iterated": True},
)

STAGE_BY_KEY = {s["key"]: s for s in STAGES}

# WebUI/static/js/docs.js: {"text"} is a paragraph, {"lead", "text"} a paragraph
# opening with a bold label, {"list", "ordered"} a list, {"formula"} a block of
# notation set in the monospace face.
#
# A section's own `blocks` are its opening, before the first subsection. The
# subsections are what the sidebar navigates to and what the N.M numbering is
# built from; their titles were the bold labels the prose used to open those
# paragraphs with, moved up into a heading now that a section is long enough to
# need navigating inside. Numbering lives in the frontend, not here, so a
# section inserted later renumbers everything after it without an edit.
#
# `gan` has no subsections: three paragraphs and no internal structure to point
# at.
DOC_SECTIONS: Tuple[Dict[str, Any], ...] = (
    {"id": "overview", "title": "Overview", "stages": (), "blocks": (
        {"text": "RAMSeS selects a detection strategy for one time series rather "
                 "than assuming one detector fits all of them. What counts as an "
                 "anomaly is context-dependent, and a detector that wins on one "
                 "series degrades on another, so the framework runs a pool of "
                 "pre-trained base detectors and decides, per series, how to use "
                 "them."},
    ), "subsections": (
        {"id": "overview-pool", "title": "The base detector pool", "blocks": (
            {"text": "By default different base detectors compete such as: "
                     "LOF_1..LOF_4, NN_1..NN_3, CBLOF_1..CBLOF_4. Each is "
                     "trained beforehand and, for every window of the series, "
                     "emits an anomaly score. Every stage below consumes those "
                     "scores; none of them retrains a base detector."},
        )},
        {"id": "overview-branches", "title": "The ensemble and single-model branches", "blocks": (
            {"text": "From the shared pool the framework runs two branches with "
                     "different aims."},
            {"list": (
                "The ensemble branch searches for a subset of detectors whose "
                "scores, stacked and fed to a meta-learner, detect better "
                "together than any of them alone.",
                "The single-model branch searches for the one detector that "
                "holds up best, combining an adaptive bandit with three "
                "independent robustness tests.",
            )},
        )},
        {"id": "overview-pipeline", "title": "The six offline sub-stages", "blocks": (
            {"text": "The offline stage runs six sub-stages: 6.1 genetic "
                     "algorithm, 6.2 Thompson sampling, 6.3 GAN perturbation, "
                     "6.4 off-by-threshold, 6.5 Monte Carlo, 6.6 rank "
                     "aggregation. The three robustness tests are independent of "
                     "one another and each works on its own copy of the data, so "
                     "no test can see another's perturbations."},
        )},
        {"id": "overview-decision",
         "title": "Two aggregations and the final decision", "blocks": (
            {"text": "Rank aggregation runs twice. First it merges the six "
                     "rankings the robustness tests produce (three tests, each "
                     "ranking by F1 and by PR-AUC) into a robustness consensus. "
                     "Then it merges that consensus with the Thompson ranking "
                     "into the final single-model order. Finally the framework "
                     "evaluates the winning ensemble and the winning single "
                     "detector on the same data and deploys whichever scores "
                     "higher on F1."},
        )},
        {"id": "overview-online", "title": "Online phase and re-optimisation", "blocks": (
            {"text": "The series is split 80% offline / 20% online. The online "
                     "portion is processed in overlapping windows, and every N "
                     "windows (default 5) the framework re-optimises, "
                     "concatenating the most recent online windows while "
                     "dropping an equal number of the oldest offline samples so "
                     "the training size stays constant. This adapts to "
                     "distribution shift without forgetting earlier behaviour. "
                     "Three strategies are available: adaptive (re-optimise "
                     "periodically), fixed-best (keep the offline winner), and "
                     "fixed-random (a baseline)."},
        )},
        {"id": "overview-explained", "title": "How the explanations are produced",
         "blocks": (
            {"text": "Every stage below can be run with an explainability layer "
                     "on top of it, disabled by default and enabled with "
                     "--explain. The layers never change what the stage decides: "
                     "they observe a run as it happens without altering it, and "
                     "each one writes a report, generates a set of figures, and "
                     "a structured record of its findings."},
            {"text": "That record is what a local language model turns into the "
                     "prose on each stage's card. It is given canonical "
                     "sentences with the numbers already computed and rounded, "
                     "never raw output, so it composes and compresses rather "
                     "than calculating or inferring. Every narrative is then "
                     "scored mechanically against that record for two things: "
                     "claims that match nothing in it, and required facts it "
                     "failed to state. Those two rates are reported per stage. "
                     "Limitations are listed separately and verbatim, never left "
                     "to the model to paraphrase."},
        )},
    )},
    {"id": "ga", "title": "Genetic algorithm",
     "stages": ("ga_selection", "ga_combination"), "blocks": (
        {"text": "The ensemble branch builds a stacking ensemble: the base "
                 "detectors are level-0 learners, and their scores become the "
                 "input features of a level-1 meta-learner that makes the final "
                 "call. The question is which subset of detectors to stack."},
    ), "subsections": (
        {"id": "ga-meta-learner",
         "title": "Why the meta-learner is fixed", "blocks": (
            {"text": "RAMSeS uses a Random Forest, chosen for F1 comparable to "
                     "an SVM at lower cost; logistic regression, gradient "
                     "boosting and SVM are also supported. It is not searched "
                     "over per subset, because doing so would raise the risk of "
                     "overfitting, destabilise the optimisation, and make "
                     "convergence harder to read. Fixing it restricts the search "
                     "to the subsets themselves."},
        )},
        {"id": "ga-loop", "title": "The search loop and its fitness", "blocks": (
            {"text": "The algorithm initialises a population of 20 randomly "
                     "drawn distinct subsets and runs for 20 generations at a "
                     "mutation rate of 0.1. In each generation it:"},
            {"ordered": True, "list": (
                "Trains the meta-learner on the stacked outputs of each "
                "candidate subset",
                "Scores each subset on a held-out validation fold",
                "Keeps the top performers as the elite pool",
                "Crosses pairs of parents drawn from that pool into new subsets",
                "Mutates some of the resulting new subsets, adding, removing or "
                "replacing a detector.",
            )},
            {"text": "Fitness is the ensemble's best-threshold F1 score, the "
                     "objective the search maximises. After the last generation "
                     "the highest-scoring subset across all generations is the "
                     "chosen ensemble, and the meta-learner trained on it is "
                     "what runs in deployment."},
        )},
        {"id": "ga-meta-use",
         "title": "From detector scores to one prediction", "blocks": (
            {"text": "For a point in time-series, every detector has an "
                     "anomaly score as output, which is in different units for "
                     "each detector. Meta-learner "
                     "gets the output of all of the level-0 detectors that are "
                     "in the subset as input and produces its own predicted "
                     "probability that the point is an anomaly. It must be "
                     "trained first, so that it learns how much to trust each "
                     "detector and in what direction. It does not average them "
                     "and it does not weight them by any external notion of "
                     "quality; the weighting is whatever fitting the training "
                     "data produced."},
        )},
        {"id": "ga-explained", "title": "What the explanation answers", "blocks": (
            {"text": "The search evaluates hundreds of subsets and reports one. "
                     "Two questions survive that: why each detector ended up in "
                     "the chosen ensemble, and how the meta-learner uses them "
                     "once it has selected them."},
        )},
        {"id": "ga-why-chosen", "title": "Utility, stability and LOFO", "blocks": (
            {"text": "Two properties are measured after the search, from the "
                     "subsets the search itself evaluated. Utility is a "
                     "detector's mean marginal contribution: the average change "
                     "in fitness between the subsets that contained the detector "
                     "and those that did not. Stability is its survival rate, "
                     "the share of evaluated subsets that detector appeared in. "
                     "Each detector is placed on both axes at once and labelled "
                     "high or low on each, split at the median of the detector "
                     "pool, so \"high utility\" means above the other detectors "
                     "here rather than good in absolute terms. A third quantity, "
                     "LOFO, is reported for members of the final ensemble only: "
                     "the ensemble's F1 score loss when that one detector is "
                     "removed from it. Utility is a global average over the "
                     "whole search, LOFO is local to the one ensemble that was "
                     "chosen, and the two can disagree in sign."},
            {"formula": "utility(d)   = mean{ F1(S) : d ∈ S } − mean{ F1(S) : d ∉ S }\n"
                        "stability(d) = (1/G) · Σ_g |{ individuals in generation g containing d }| / P\n"
                        "LOFO(d)      = F1(Ŝ) − F1(Ŝ \\ {d}),    d ∈ Ŝ"},
            {"text": "where S ranges over the distinct subsets the GA evaluated, "
                     "Ŝ is the chosen ensemble, G is the number of generations "
                     "and P the size of the population in each generation."},
        )},
        {"id": "ga-meta-explained",
         "title": "SHAP, PFI and ALE", "blocks": (
            {"text": "Three measures each rank the chosen detectors by how much "
                     "weight they carry, and each answers a different question. "
                     "SHAP measures how much the meta-learner's anomaly "
                     "probability moves when a detector's actual output is "
                     "revealed in place of its average output, averaged over "
                     "every combination of the other detectors being revealed or "
                     "held at their averages. Because the ensemble is small, "
                     "every combination is enumerated exactly rather than "
                     "sampled. PFI measures how far F1 falls when a detector's "
                     "score column is shuffled, so unlike SHAP it uses the "
                     "labels and reports reliance on the detector rather than "
                     "influence on the output. ALE sweeps a detector across its "
                     "own observed score range in narrow bands and accumulates "
                     "how far the meta-learner's output moves, using only the "
                     "rows that fall in each band."},
            {"text": "The three measures are magnitudes. Because they might "
                     "disagree, they are merged by the same kind of Markov rank "
                     "aggregation the single-model branch uses, giving one "
                     "overall weight ranking."},
        )},
        {"id": "ga-direction", "title": "The sign and how well it is supported",
         "blocks": (
            {"text": "A detector's sign is the direction of ALE's total "
                     "accumulated effect: positive if a rising score pushes the "
                     "meta-learner toward flagging the point as an anomaly, "
                     "negative if towards flagging normal. How well that sign is "
                     "supported is reported separately, because two things can "
                     "undermine it: an effect that changes direction across the "
                     "score range, or a detector that barely moves the "
                     "meta-learner at all. Both are named rather than allowed to "
                     "suppress the sign, since a blank reads as missing data "
                     "when the measurement was actually made."},
        )},
    )},
    {"id": "lints", "title": "Linear Thompson Sampling",
     "stages": ("thompson_ranking", "thompson_sampling"), "blocks": (
        {"text": "LinTS treats detector choice as a contextual bandit. Each "
                 "detector is an arm; each window of the series is a round; "
                 "pulling an arm means running that detector on that window and "
                 "observing how well it did. The time-series is split into "
                 "windows in the beginning."},
    ), "subsections": (
        {"id": "lints-bayesian", "title": "The context vector and the posterior", "blocks": (
            {"text": "Every window is turned into a context vector x (its "
                     "channel readings over the window's timesteps). Each "
                     "detector holds a Bayesian linear model of its own reward, "
                     "E[r | x] = θᵀx, with a Gaussian posterior over θ "
                     "summarised by a mean vector μ and covariance Σ. The "
                     "posterior starts at μ = 0 and Σ = I, a ridge prior that "
                     "keeps early updates stable."},
        )},
        {"id": "lints-round", "title": "Choosing a detector for a window", "blocks": (
            {"text": "With probability ε the framework picks a detector "
                     "uniformly at random. Otherwise it draws one sample θ̃ from "
                     "every detector's posterior and picks the detector "
                     "maximising θ̃ᵀx. Sampling rather than taking the mean is "
                     "what makes the choice uncertainty-aware: a detector that "
                     "has rarely been tried has a wide posterior and can win on "
                     "a favourable draw, so the run keeps testing plausible "
                     "alternatives instead of locking onto an early leader."},
        )},
        {"id": "lints-reward", "title": "The reward and the posterior update", "blocks": (
            {"text": "The chosen detector is evaluated on that window based on "
                     "injected timesteps and rewarded with a weighted "
                     "combination of F1 and PR-AUC (default 0.5 each). Only the "
                     "chosen detector's posterior is updated, by Bayesian linear "
                     "regression on the pair (x, r): the covariance absorbs xxᵀ "
                     "and the mean moves toward the observed reward."},
        )},
        {"id": "lints-epsilon", "title": "How exploration decays", "blocks": (
            {"text": "ε starts at 0.2 and is annealed after every window, so the "
                     "run begins broad and narrows toward exploiting what it has "
                     "learned."},
        )},
        {"id": "lints-output", "title": "The final ranking by ‖μ‖²", "blocks": (
            {"text": "The offline phase runs until every window is processed. "
                     "Afterwards the detectors are ranked by the overall size of "
                     "their learned weights, ‖μ‖². This ranking is the branch's "
                     "output and is what goes into the final aggregation."},
        )},
        {"id": "lints-two-views", "title": "μᵀx versus ‖μ‖²", "blocks": (
            {"text": "The branch produces two different numbers, which is why it "
                     "has two cards. μᵀx is the expected reward at a given "
                     "window and is what drove the choice made there; it moves "
                     "as the series moves. ‖μ‖² is the accumulated ranking score "
                     "and only grows when a detector is picked and rewarded. A "
                     "detector can therefore lead the expected-reward view for "
                     "much of the run and still not finish first in the "
                     "ranking."},
        )},
        {"id": "lints-note", "title": "Why contexts are normalised", "blocks": (
            {"text": "Context vectors are normalised to unit length before use. "
                     "Without it, a series with large sensor values or many "
                     "channels (SMD carries 38) makes xxᵀ dominate the "
                     "covariance update and collapse Σ."},
        )},
        {"id": "lints-explained", "title": "What the two cards explain",
         "blocks": (
            {"text": "The branch produces two quantities and both are explained, "
                     "which is why it has two cards."},
        )},
        {"id": "lints-criterion", "title": "Splitting the score across channels", "blocks": (
            {"text": "The score ‖μ‖² is a sum of squared weights, so it splits "
                     "exactly into one number per channel with no approximation: "
                     "a channel's contribution to a detector is the sum of the "
                     "squares of that detector's mean-vector weights for that "
                     "channel, and the contributions add up to the whole score. "
                     "Its share is that contribution as a fraction of the score. "
                     "Because these are squares they are never negative, so a "
                     "small share means a channel added little, not that it "
                     "worked against the detector. Comparing two detectors does "
                     "have direction: the difference of their contributions "
                     "channel by channel sums exactly to the margin (difference) "
                     "between their scores, and a negative term marks a channel "
                     "the rival was stronger on. The stage also reports how "
                     "often each detector was actually selected, because μ only "
                     "moves in windows where the arm was pulled, so the score "
                     "reflects exposure as well as quality. Regimes here are "
                     "stretches where one detector held the highest score, read "
                     "straight off the leader at each window after a short "
                     "warm-up in which every score is still zero."},
            {"formula":
                "contribution(k, c)     = Σ_{i ∈ channel c} μ_k[i]²\n"
                "Σ_c contribution(k, c) = ‖μ_k‖²\n"
                "margin(a, b, c)        = contribution(a, c) − contribution(b, c)\n"
                "Σ_c margin(a, b, c)    = ‖μ_a‖² − ‖μ_b‖²"},
            {"text": "where k, a and b are detectors, c is a channel, and i runs "
                     "over the entries of that detector's mean vector μ that "
                     "belong to channel c."},
        )},
        {"id": "lints-dynamics", "title": "The reward split and the selection states", "blocks": (
            {"text": "The expected reward μᵀx splits per channel the same way, "
                     "and those contributions sum to the prediction exactly. A "
                     "second and narrower measure is also reported: SHAP, how "
                     "far a channel's contribution departs from what that "
                     "channel on average contributes, measured against the "
                     "average window of the run. The two answer different "
                     "questions, and a channel can supply most of a detector's "
                     "reward while departing from its own norm not at all. "
                     "Regimes here are stretches of at least three consecutive "
                     "windows in which one detector held the highest expected "
                     "reward, computed on the beliefs held before each window's "
                     "update so a regime describes the decision that was made "
                     "rather than its aftermath. Shorter changes of lead are "
                     "recorded as blips and not treated as regimes. This regime "
                     "is different from the previous one, so the both need not "
                     "line up. Every choice the sampler made is also classified "
                     "as exploitation, informed exploration or random "
                     "exploration, so the run can be read as behaviour and not "
                     "only as an outcome. Exploitation: The sampler picked the "
                     "detector with the highest μᵀx. Informed exploration: The "
                     "sample drawn from the posterior led to a different "
                     "detector than the one with the highest μᵀx. Random "
                     "exploration: A forced exploration step fired, so the pick "
                     "was random rather than informed. A run that is mostly "
                     "random exploration is one where ε had not yet decayed; one "
                     "that is mostly informed exploration is one where the "
                     "detectors stayed closely matched and the posteriors "
                     "uncertain."},
            {"formula":
                "contribution(k, c)     = Σ_{i ∈ channel c} μ_k[i] · x[i]\n"
                "Σ_c contribution(k, c) = μ_kᵀx"},
            {"text": "where k is a detector, c is a channel, x is the window's "
                     "context vector, and i runs over the entries belonging to "
                     "channel c."},
        )},
    )},
    {"id": "gan", "title": "GAN perturbation test", "stages": (), "blocks": (
        {"text": "GAN test asks whether a detector has learned the structure of "
                 "normality or merely memorised its training data. A generator "
                 "and a discriminator, each a two-layer MLP with 256 hidden "
                 "units, ReLU activations and dropout, are trained adversarially "
                 "on the clean, un-augmented split, with label smoothing and "
                 "added noise for stability. Training on clean data only is what "
                 "keeps the test from leaking into the detectors it later "
                 "judges."},
        {"text": "Once trained, the generator produces a pool of candidate "
                 "points, the discriminator scores them, and the framework keeps "
                 "the most ambiguous ones: those closest to the discriminator's "
                 "decision threshold. These are the borderline cases, plausible "
                 "enough to belong to the series but sitting where \"normal\" "
                 "and \"anomalous\" are hardest to separate. The discriminator's "
                 "own verdict supplies each injected point's label."},
        {"text": "The selected points are interleaved into the stream at regular "
                 "intervals so chronology is preserved, at an injection budget "
                 "of about 10% of the original samples. Every detector is then "
                 "re-evaluated on the augmented series and ranked by F1 and by "
                 "PR-AUC. Those two rankings are among the six that feed the "
                 "robustness consensus."},
    ), "subsections": ()},
    {"id": "off-by", "title": "Off-by-threshold test",
     "stages": ("off_by_threshold",), "blocks": (
        {"text": "Off-by-threshold tests ask where a detector draws the line "
                 "between \"just unusual\" and \"anomalous\", by manufacturing "
                 "points that sit almost exactly on the decision threshold."},
    ), "subsections": (
        {"id": "off-by-idea", "title": "Why borderline points are the hard case", "blocks": (
            {"text": "A point far outside the normal range is easier to detect. "
                     "A point barely outside of the usual series behaviour is "
                     "the hard case, and it is also the common case. This test "
                     "builds those points deliberately, with known labels, and "
                     "checks which detectors classify them correctly."},
        )},
        {"id": "off-by-build", "title": "Building a point from local variation", "blocks": (
            {"text": "About one point is inserted for every ten already in the "
                     "series, at evenly spaced positions. To build the point at "
                     "a given position, the framework looks at a short stretch "
                     "of the series around it and measures how much each channel "
                     "varies there, its local standard deviation. That number is "
                     "the series' own account of how much variation is ordinary "
                     "at this spot, and the injected point is drawn at roughly "
                     "that size, multiplied by a random factor drawn just above "
                     "or just below one."},
            {"text": "Scaling by local variation rather than by a fixed amount "
                     "is what makes this a boundary test. A fixed absolute "
                     "deviation would be unremarkable in a noisy stretch and "
                     "glaring in a quiet one, so a fixed perturbation would be "
                     "trivial to detect in some places and invisible in others. "
                     "Using the local variation puts every injected point at the "
                     "same relative distance from normal, independent from the "
                     "position in time-series."},
        )},
        {"id": "off-by-label", "title": "How the random factor sets the label", "blocks": (
            {"text": "The random factor decides. At or below 1, the point varies "
                     "no more than its neighbourhood does and is labelled "
                     "normal. Above 1, it exceeds the local norm and is labelled "
                     "a statistical border anomaly. Because the factor only ever "
                     "lands slightly either side of one, both classes sit right "
                     "at the boundary, which is the whole point of the test."},
        )},
        {"id": "off-by-measured", "title": "Re-ranking on the augmented series", "blocks": (
            {"text": "Every detector is re-evaluated on the augmented series and "
                     "ranked by F1 and by PR-AUC, contributing two of the six "
                     "rankings. The augmented data is used nowhere else in the "
                     "framework. The test needs at least 100 points and both "
                     "classes present in the original labels, and is skipped "
                     "rather than run on data that cannot support it."},
        )},
        {"id": "off-by-explained", "title": "Exclusive wins and the surrogate trees",
         "blocks": (
            {"text": "The perturbation here happens at the level of individual "
                     "points, so the explanation does too. Rather than "
                     "re-running the test under different settings, it reuses "
                     "the single production run and asks, per injected point, "
                     "what kind of borderline point the winner handles that a "
                     "given rival does not."},
            {"text": "For the winning detector and each other detector in turn, "
                     "an exclusive win is an injected point the winner "
                     "classified correctly and that rival did not. A small "
                     "decision tree is fitted to predict those exclusive wins "
                     "from four properties of the point, none of which depends "
                     "on any detector: its boundary distance, how far the random "
                     "factor landed from 1 and therefore how far off the "
                     "threshold the point sits; whether it was injected as an "
                     "anomaly or as normal; its local volatility, the local "
                     "standard deviation at the injection site; and its position "
                     "in the series, from 0 at the start to 1 at the end. The "
                     "tree's splits become plain rules, and the average "
                     "importance across all the rival trees shows which property "
                     "best explains the winner's edge."},
            {"text": "Each tree also reports a held-out, cross-validated "
                     "accuracy alongside the accuracy on the points it was "
                     "fitted to, because the second can look strong purely from "
                     "memorising a small set of wins. A comparison resting on "
                     "fewer exclusive wins than there are cross-validation folds "
                     "is flagged, since the held-out estimate is not stable "
                     "there. Only the prediction side is explained: correctness "
                     "is defined by thresholded predictions, and PR-AUC has no "
                     "per-point notion of right or wrong."},
        )},
    )},
    {"id": "monte-carlo", "title": "Monte Carlo simulation",
     "stages": ("monte_carlo",), "blocks": (
        {"text": "In Monte Carlo, zero mean Gaussian noise is added to every "
                 "channel at every timestep, and every detector is re-evaluated "
                 "on an independent draw of that noise. The labels do not "
                 "change, the ground truth stays exactly as it was and only the "
                 "signal degrades, so this measures whether a detector's "
                 "standing survives a dirtier version of the same series. Scores "
                 "are averaged across trials, so a detector that happens to win "
                 "one trial does not carry the ranking. The averaged F1 and "
                 "PR-AUC orderings are the last two of the six rankings entering "
                 "the robustness consensus."},
        {"text": "The noise level is the standard deviation of the injected "
                 "Gaussian noise, fixed for the production ranking."},
    ), "subsections": (
        {"id": "mc-explained", "title": "The noise sweep and its two views", "blocks": (
            {"text": "The production test holds the noise level fixed, and at "
                     "one fixed level independent trials carry no structure to "
                     "explain. The explainability layer therefore sweeps that "
                     "one parameter, running the same noise injection across a "
                     "range of levels and repeating each several times, to show "
                     "which detector wins under how much noise. This sweep is "
                     "entirely separate: the production ranking that feeds rank "
                     "aggregation is untouched."},
            {"text": "Two views are produced over that sweep. The first is the "
                     "score of every detector as noise grows, marking the "
                     "crossovers where the lead changes hands. The second is a "
                     "small decision tree fitted from noise level to winning "
                     "detector, which turns those crossovers into explicit "
                     "thresholds, together with per-detector trends summarising "
                     "how steeply each degrades."},
        )},
        {"id": "mc-f1", "title": "Adaptive versus frozen thresholds", "blocks": (
            {"text": "Re-choosing each detector's best threshold at every noise "
                     "level measures the best it could possibly do, which hides "
                     "the degradation a deployed detector would actually suffer, "
                     "because in production one threshold is committed to. So "
                     "the sweep also freezes each detector's threshold at the "
                     "lowest noise level and holds it across the sweep. The gap "
                     "between the two at high noise is the erosion the first "
                     "view conceals."},
        )},
        {"id": "mc-not", "title": "Limits of the sweep", "blocks": (
            {"text": "It scores with a fast point-wise metric rather than the "
                     "range-based one the production ranking uses, so its "
                     "numbers are not directly comparable to production values, "
                     "and it exists to show shape rather than to rank. And the "
                     "surrogates report a held-out, cross-validated fidelity "
                     "next to their fit on the sweep itself; on flat curves cut "
                     "into small folds the held-out figure can legitimately be "
                     "worse than predicting the mean, and where most folds are "
                     "degenerate the estimate is marked unavailable rather than "
                     "reported as a number."},
        )},
    )},
    {"id": "rank-aggregation", "title": "Rank aggregation",
     "stages": ("rank_aggregation_robust", "rank_aggregation_final"), "blocks": (
        {"text": "The single-model branch produces seven orderings of the same "
                 "detectors: six from the robustness tests and one from Thompson "
                 "sampling. They may disagree, and each is about a different "
                 "thing, so none can simply be preferred. Aggregation turns them "
                 "into one consensus."},
    ), "subsections": (
        {"id": "agg-method", "title": "From pairwise counts to a consensus", "blocks": (
            {"text": "For every pair of detectors, count how many of the input "
                     "rankings put i ahead of j. Normalise those counts into a "
                     "row-stochastic transition matrix, giving a random walk "
                     "over detectors that is more likely to move toward a "
                     "detector that beats the current one more often. Compute "
                     "the stationary distribution of that walk by power "
                     "iteration, and sort the detectors by it. A detector scores "
                     "highly when the detectors that beat it are themselves "
                     "rarely preferred, so the result reflects the whole "
                     "structure of agreement rather than a count of first "
                     "places."},
        )},
        {"id": "agg-twice", "title": "The two aggregation stages", "blocks": (
            {"text": "First to the six robustness rankings, producing the "
                     "robustness consensus. Then to that consensus together with "
                     "the Thompson ranking, producing the final single-model "
                     "order. The two-stage shape is deliberate: it keeps the "
                     "three robustness tests from outvoting the adaptive branch "
                     "six to one."},
        )},
        {"id": "agg-explained", "title": "Influence, agreement and Borda", "blocks": (
            {"text": "The consensus is one ordering built from several possibly "
                     "disagreeing ones, and the question the explanation answers "
                     "is which sources shaped it the most."},
            {"text": "Each source gets two scores. Its influence is how far the "
                     "consensus moves when that source is dropped and the "
                     "aggregation is re-run without it. Its agreement is how "
                     "closely the source's own ranking matches the consensus "
                     "that was produced. The two are independent, and the "
                     "interesting cases are where they diverge."},
            {"formula": "influence(s)  = (1 − τ(R, R₋ₛ)) / 2        ∈ [0, 1]\n"
                        "agreement(s)  = τ(rₛ, R)                    ∈ [−1, 1]"},
            {"text": "where R is the consensus, R₋ₛ the consensus re-aggregated "
                     "without source s, rₛ that source's own ranking, and τ is "
                     "Kendall's tau."},
            {"text": "Each score induces its own ranking of the sources, and "
                     "those two rankings are then treated as two voters and "
                     "merged by Borda count into a single standing. That "
                     "standing is the verdict, and it is what \"shaped the "
                     "consensus most, second most, and so on\" reports. "
                     "Comparing a source's two ranks also gives it a pattern: an "
                     "influential disagreer moved the consensus a long way while "
                     "agreeing with it poorly, meaning it pulled the result away "
                     "from its own view and was partly counteracted; a redundant "
                     "agreer agrees closely but changed nothing, its view "
                     "already covered by others; a consistent source ranks "
                     "similarly on both."},
            {"formula": "borda(s) = (N − rank_influence(s)) + (N − rank_agreement(s))"},
            {"text": "where N is the number of ranking sources being merged."},
        )},
        {"id": "agg-two-source", "title": "Why the final stage uses agreement alone",
         "blocks": (
            {"text": "Where only two rankings are merged, for example in the "
                     "final aggregation, leave-one-out is degenerate, since "
                     "dropping one leaves the other unchanged, and Borda voting "
                     "over two sources says nothing. The final aggregation is "
                     "therefore explained by agreement alone: which of the two "
                     "the consensus leans toward, and by how much."},
        )},
    )},
)

DOC_SECTION_BY_STAGE = {key: section["id"]
                        for section in DOC_SECTIONS for key in section["stages"]}

# One line per term, shown on the stage card in the space the glossary used to
# fill. Ordered pairs rather than a dict so the reader meets the terms in the
# order the card's prose uses them, not alphabetically.
#
# These define the stage's vocabulary and are the same for every run, which is
# why they live here rather than in the IR: nothing about them depends on what a
# run found. The long-form glossary on /docs stays where it is, written by
# `Explainability.ir` into each stage's artifacts.
STAGE_TERMS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "ga_selection": (
        ("Utility", "The average change in an ensemble's F1 score when the "
                    "detector is added to it."),
        ("Stability", "The share of the subsets the algorithm evaluated that "
                      "included the detector."),
    ),
    "ga_combination": (
        ("SHAP", "How much the meta-learner's anomaly probability moves when a "
                 "detector's actual output is revealed in place of its average "
                 "output."),
        ("PFI", "How far F1 drops when that detector's scores are shuffled; "
                "label-based."),
        ("ALE", "The change in the meta-learner's anomaly probability as the "
                "detector's own score is gradually increased."),
        ("Sign", "The direction of that change: positive means the probability "
                 "rises, negative means it falls."),
    ),
    "thompson_ranking": (
        ("Score", "The overall size ‖μ‖² of a detector's learned mean vector μ."),
        ("Share", "The fraction of that score that came from a single channel."),
        ("Contribution", "The exact amount a single channel added to that score."),
        ("Margin", "The gap between two detectors' scores; it traces back to each "
                   "channel, whose contributions sum to it exactly."),
        ("Regime", "A stretch of windows in which one detector held the highest "
                   "score ‖μ‖²."),
    ),
    "thompson_sampling": (
        ("Expected reward", "A detector's predicted reward μᵀx for a window, its "
                            "weights applied to that window's data."),
        ("Regime", "At least three consecutive windows in which one detector held "
                   "the highest expected reward."),
        ("Exploitation", "The sampler picked the detector with the highest μᵀx."),
        ("Informed exploration", "The sample drawn from the posterior led to a "
                                 "different detector than the one with the "
                                 "highest μᵀx."),
        ("Random exploration", "A forced exploration step fired, so the pick was "
                               "random rather than informed."),
        ("SHAP", "How far a channel's contribution to the expected reward "
                 "departed from its average contribution over the run."),
    ),
    "monte_carlo": (
        ("Noise level", "The standard deviation of the Gaussian noise injected "
                        "into the data."),
    ),
    "off_by_threshold": (
        ("Exclusive win", "A point the top-ranked detector classified correctly "
                          "and the named rival did not."),
        ("Boundary distance", "How far the point was scaled away from the "
                              "decision boundary; 0 sits exactly on it."),
        ("Local volatility", "The standard deviation of the series around the "
                             "injection site; how noisy that neighbourhood is."),
        ("Position", "Where the point falls in the series, from 0 at the start "
                     "to 1 at the end."),
    ),
    # Both consensus cards get all three: they are one stage, and the vocabulary
    # is the stage's. A two-source aggregation reports no influence — leave-one-
    # out is undefined there — but defining the word costs nothing and the card's
    # own prose is what says whether it applied.
    "rank_aggregation_robust": (
        ("Influence", "How much the aggregated ranking changes when that source "
                      "is left out."),
        ("Agreement", "How similar the source's own ranking is to the aggregated "
                      "ranking."),
        ("Pattern", "The two read together: influential disagreer, redundant "
                    "agreer, or consistent."),
    ),
}
STAGE_TERMS["rank_aggregation_final"] = STAGE_TERMS["rank_aggregation_robust"]

# The GAN test runs but has no explainability layer; the global IR records that
# explicitly. Surfacing it as a stage with a reason beats an unexplained gap.
GAN_STAGE = {"key": "gan", "title": "Robustness: GAN perturbations",
             "cli": "gan", "plot_group": "gan", "order": 9}


def split_info(raw: str) -> Tuple[Optional[str], str]:
    """`"INFO: glossary\\n\\nnarrative"` -> `("glossary", "narrative")`.

    The glossary leads the file (Explainability/llm.py writes it before the
    narrative). Anything that does not start with the marker is all narrative.
    """
    if raw is None:
        return None, ""
    text = raw.replace("\r\n", "\n").strip("\n")
    if not text.startswith("INFO:"):
        return None, text.strip()
    body = text[len("INFO:"):]
    info, sep, narrative = body.partition("\n\n")
    if not sep:
        # Glossary with no narrative after it.
        return info.strip(), ""
    return info.strip(), narrative.strip()


def _read_text(path: Path) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _newest(pattern_dir: Path, pattern: str) -> Optional[Path]:
    """Newest file matching `pattern` inside `pattern_dir`, by mtime.

    Iteration numbers are NOT consistent across artifact trees: the
    comprehensive report uses the CLI --iteration (default 5) while the
    explanations use OFFLINE_ITERATION = 0, and both can coexist in one
    directory. Always glob and take the newest; never derive the index.
    """
    matches = glob.glob(os.path.join(glob.escape(str(pattern_dir)), pattern))
    if not matches:
        return None
    return Path(max(matches, key=lambda p: os.path.getmtime(p)))


# A narrative is stale once its IR has been rewritten under it. This happens
# for real: `--stages X --explain` is a PARTIAL run, and app.py returns before
# the narration block, so it regenerates the IR and every plot but leaves the
# prose alone. The page then shows fresh figures beside sentences describing a
# run that no longer exists — on SKAB/7 the narrative walked 14 regimes while
# the IR and the plots had 10. Nothing else notices, because the two files are
# read independently. One second of slack absorbs same-run write ordering.
_STALE_SLACK_SECONDS = 1.0


def _mtime(path: Optional[Path]) -> Optional[float]:
    try:
        return path.stat().st_mtime if path is not None else None
    except OSError:
        return None


def _load_stage_files(ir_dir: Optional[Path], nl_dir: Optional[Path],
                      stage: Dict[str, Any]) -> Tuple[Optional[dict], Optional[str],
                                                      Optional[Path], bool]:
    """(ir_doc, raw_nl_text, nl_path, narrative_is_stale) for one stage."""
    ir_doc = raw = None
    ir_path = nl_path = None
    if ir_dir is not None:
        exact = ir_dir / f"{stage['ir']}.json"
        ir_path = exact if exact.exists() else _newest(ir_dir, f"{stage['ir']}*.json")
        if ir_path is not None:
            ir_doc = _read_json(ir_path)
    if nl_dir is not None:
        exact = nl_dir / f"{stage['nl']}.txt"
        nl_path = exact if exact.exists() else _newest(nl_dir, f"{stage['nl']}*.txt")
        if nl_path is not None:
            raw = _read_text(nl_path)
    ir_at, nl_at = _mtime(ir_path), _mtime(nl_path)
    stale = bool(ir_at and nl_at and nl_at + _STALE_SLACK_SECONDS < ir_at)
    return ir_doc, raw, nl_path, stale


def load_global_ir(dataset: str, entity: str) -> Optional[dict]:
    ir_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_IR, dataset, entity)
    if ir_dir is None:
        return None
    path = _newest(ir_dir, "ir_global_iter*.json")
    return _read_json(path) if path else None


def load_faithfulness(dataset: str, entity: str) -> Optional[dict]:
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if nl_dir is None:
        return None
    path = _newest(nl_dir, "faithfulness_iter*.json")
    return _read_json(path) if path else None


def global_text_path(dataset: str, entity: str) -> Optional[Path]:
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if nl_dir is None:
        return None
    return _newest(nl_dir, "nl_global_iter*.txt")


_ITER_RE = re.compile(r"_iter(\d+)\.txt$")


def comprehensive_path(dataset: str, entity: str) -> Optional[Path]:
    """Newest `comprehensive_results_*.txt` for this entity, or None.

    Written by the pipeline itself, not by the explainability layer, so it
    exists for runs made without `--explain` and is absent after a partial run
    (`app.py` returns before the report step). Its iteration index comes from
    `--iteration` (default 5) rather than the explanations' OFFLINE_ITERATION,
    which is exactly why this globs instead of building the filename.
    """
    report_dir = paths.resolve_entity_dir(paths.COMPREHENSIVE, dataset, entity)
    if report_dir is None:
        return None
    return _newest(report_dir, "comprehensive_results_*.txt")


def comprehensive_info(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """Metadata for the report, without reading it — the page links, not inlines."""
    path = comprehensive_path(dataset, entity)
    if path is None:
        return None
    match = _ITER_RE.search(path.name)
    try:
        stat = path.stat()
    except OSError:
        return None
    return {
        "name": path.name,
        "iteration": int(match.group(1)) if match else None,
        "bytes": stat.st_size,
        "generated_at": stat.st_mtime,
        "url": f"/report/{dataset}/{entity}",
        "download_url": f"/api/comprehensive/{dataset}/{entity}?download=1",
    }


def comprehensive_report(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """`comprehensive_info` plus the report text itself."""
    info = comprehensive_info(dataset, entity)
    if info is None:
        return None
    path = comprehensive_path(dataset, entity)
    return {**info, "text": _read_text(path) or "",
            "dataset_label": dataset_label(dataset)}


# Matches both Thompson stages' regime atoms — `ts.regime.N` (expected-reward
# regimes) and `tsr.regime.N` (||mu||^2 leadership regimes). Anchored on the
# suffix rather than the prefix so a third producer needs no change here.
_REGIME_RE = re.compile(r"\.regime\.(\d+)$")


def _regimes_from_ir(ir_doc: dict) -> List[Dict[str, Any]]:
    """Regime atoms, ordered, ready to pair with their per-regime plots.

    The ids are 0-based and match `regime_{NN}_w{start}-{end}_{model}.png`
    exactly, so each regime sentence can be shown beside its own plot instead of
    the reader hunting through fourteen images.
    """
    out = []
    for atom in ir_doc.get("evidence", []) or []:
        m = _REGIME_RE.search(str(atom.get("id", "")))
        if not m:
            continue
        value = atom.get("value") or {}
        out.append({
            "index": int(m.group(1)),
            "start": value.get("start"),
            "end": value.get("end"),
            "duration": value.get("duration"),
            "leader": value.get("leader") or atom.get("subject"),
            "text": atom.get("text", ""),
        })
    return sorted(out, key=lambda r: r["index"])


def _attach_narrated_regimes(regimes: List[Dict[str, Any]], narrative: str,
                             ir_doc: dict) -> None:
    """Give each regime the sentence the model wrote about it.

    The per-regime disclosure used to show the IR's own atom text — correct but
    flat, and a second rendering of facts the narrative already covers. The
    narrated sentences are pulled out of the same paragraph the summary drops,
    so nothing is generated twice and nothing is lost by hiding them from the
    default view. `text` stays as the fallback when a regime's sentence cannot
    be located.
    """
    by_index: Dict[int, str] = {}
    for sentence, atom in attribute_sentences(narrative, ir_doc):
        if not atom or atom.get("type") != "regime":
            continue
        m = _REGIME_RE.search(str(atom.get("id", "")))
        if m:
            idx = int(m.group(1))
            # A regime can span two sentences; keep them in narrative order.
            by_index[idx] = (by_index.get(idx, "") + " " + sentence.strip()).strip()
    for regime in regimes:
        narrated = by_index.get(regime.get("index"))
        if narrated:
            regime["narrated"] = narrated


def _headline_pick(output: Dict[str, Any]) -> Optional[str]:
    """The one detector a stage put first, whatever the stage calls that key."""
    for key in ("top_pick", "winner", "top_pick_f1"):
        value = output.get(key)
        if isinstance(value, str) and value and value != "not_available":
            return value
    return None


def _stage_faithfulness(report: Optional[dict], key: str) -> Optional[dict]:
    if not report:
        return None
    entry = (report.get("stages") or {}).get(key)
    if not entry:
        return None
    verify = entry.get("verify") or {}
    return {
        "status": entry.get("status"),
        "words": entry.get("words"),
        "hallucination_rate": verify.get("hallucination_rate"),
        "omission_rate": verify.get("omission_rate"),
        "n_claims": verify.get("n_claims"),
        "n_required": verify.get("n_required"),
        "repaired": bool(entry.get("repaired")),
    }


def build_payload(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """The `/api/explanations/<ds>/<ent>` response, or None if nothing exists."""
    ir_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_IR, dataset, entity)
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if ir_dir is None and nl_dir is None:
        return None

    global_ir = load_global_ir(dataset, entity)
    faith = load_faithfulness(dataset, entity)
    gtext = global_text_path(dataset, entity)

    stages_out: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for stage in STAGES:
        ir_doc, raw, nl_path, stale = _load_stage_files(ir_dir, nl_dir, stage)
        if ir_doc is None and raw is None:
            continue
        info, narrative = split_info(raw or "")
        summary = summarize(narrative, stage=stage["key"], ir_doc=ir_doc)
        output = (ir_doc or {}).get("output") or {}
        entry: Dict[str, Any] = {
            "key": stage["key"],
            "title": stage["title"],
            "order": stage["order"],
            "status": "ok" if narrative else "no_narrative",
            # Stages name their headline result differently: `top_pick` for
            # most, `winner` for off-by, `top_pick_f1` for Monte Carlo (which
            # ranks on two metrics). GA selection has no single pick — it
            # chooses a set — so None there is correct, not a gap.
            "top_pick": _headline_pick(output),
            "summary": summary["summary"],
            "summary_is_full": summary["is_full"],
            "summary_mode": summary["mode"],
            "summary_table": summary.get("table"),
            "extended_in": summary.get("extended_in"),
            # What the full-text disclosure shows. Usually the whole narrative;
            # a stage that renders some of its sentences elsewhere on the card
            # (Thompson's regime walk, beside its per-regime plots) hands back a
            # trimmed body so the page never prints them twice. `words` and the
            # download stay on the real narrative — the file on disk is the
            # verbatim record, and the length is what the model actually wrote.
            # `body` is that narrative minus any sentence restating a caveat:
            # the card lists the caveats verbatim below, so the disclosure must
            # not be a second, looser copy of them.
            "full": summary.get("extended") or summary.get("body") or narrative,
            "words": len(narrative.split()) if narrative else 0,
            # The glossary itself now lives on the documentation page; the card
            # keeps the one-line definitions and a pointer to the section
            # holding the long form.
            "info": info,
            "terms": [list(pair) for pair in STAGE_TERMS.get(stage["key"], ())],
            "doc_section": DOC_SECTION_BY_STAGE.get(stage["key"]),
            "question": (ir_doc or {}).get("question"),
            "output": output,
            "caveats": [c.get("text") for c in ((ir_doc or {}).get("caveats") or [])],
            "faithfulness": _stage_faithfulness(faith, stage["key"]),
            "plot_group": stage["plot_group"],
            "nl_file": nl_path.name if nl_path else None,
            # The narrative predates its own IR: the numbers on this card
            # may describe a previous run. Surfaced rather than silently
            # rendered, and cleared by re-running the narrator.
            "stale": stale,
        }
        if stage.get("regimes") and ir_doc:
            entry["regimes"] = _regimes_from_ir(ir_doc)
            _attach_narrated_regimes(entry["regimes"], narrative, ir_doc)
        stages_out.append(entry)

    # Stages the global IR knows about but that produced no narrative — GAN
    # always, plus anything a partial run skipped. Carry the IR's own note so
    # the UI states a reason instead of showing an unexplained gap.
    present = {s["key"] for s in stages_out}
    for key, info_block in sorted(((global_ir or {}).get("stages") or {}).items()):
        if key in present:
            continue
        meta = STAGE_BY_KEY.get(key) or (GAN_STAGE if key == "gan" else None)
        missing.append({
            "key": key,
            "title": (meta or {}).get("title", key.replace("_", " ").capitalize()),
            "plot_group": (meta or {}).get("plot_group"),
            "status": info_block.get("status", "not_available"),
            "note": info_block.get("note"),
        })

    decision = (global_ir or {}).get("decision") or {}
    # The final consensus IS the source of the single-model pick, so comparing
    # them always reports agreement and says nothing. Newer runs no longer emit
    # the row; filtering here means older result trees read correctly too.
    agreement = [
        {"source": name,
         "top_pick": info.get("top_pick"),
         # Absent on result trees written before rankings were carried; the
         # frontend simply renders no disclosure for those.
         "ranking": list(info.get("ranking") or []),
         "agrees": info.get("agrees_with_final_single")}
        for name, info in sorted(((global_ir or {}).get("stage_agreement") or {}).items())
        if name != "final_consensus"
    ]
    decision_atom = next(
        (a.get("text") for a in ((global_ir or {}).get("evidence") or [])
         if a.get("id") == "global.decision"), None)

    return {
        "dataset": dataset,
        # The page header shows this rather than the directory name: the run
        # tree is keyed "Anomaly_Archive" where every reader calls it UCR.
        "dataset_label": dataset_label(dataset),
        "entity": entity,
        "iteration": (global_ir or {}).get("iteration"),
        # No global IR but a global .txt on disk means an older result tree:
        # serve the raw text, never try to parse it.
        "degraded": global_ir is None,
        "decision": decision,
        "decision_text": decision_atom,
        "agreement": agreement,
        "stages": sorted(stages_out, key=lambda s: s["order"]),
        "missing_stages": missing,
        "faithfulness": (faith or {}).get("overall"),
        "model": (faith or {}).get("model"),
        "global_text": gtext.name if gtext else None,
        "generated_at": (os.path.getmtime(gtext) if gtext else None),
        # Kept beside the explanation but never merged into it: the report is
        # the pipeline's own record of what happened, in its own numbers.
        "comprehensive": comprehensive_info(dataset, entity),
    }


def documentation(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """The stage documentation: one section per pipeline stage.

    The text describes RAMSeS itself and is the same for every run, so it is
    served straight from DOC_SECTIONS. The route stays per-entity because the
    page carries a link back to the entity the reader came from, and because
    the stage cards address it that way.

    Returns None when the entity has no explanations at all, which is the one
    thing that would make that back-link point at nothing.
    """
    ir_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_IR, dataset, entity)
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if ir_dir is None and nl_dir is None:
        return None
    def _blocks(source):
        # Lists and formulas are tuples in the registry; JSON wants arrays, and
        # a copy keeps a caller from mutating the module-level text.
        out = []
        for block in source:
            item = dict(block)
            if "list" in item:
                item["list"] = list(item["list"])
            out.append(item)
        return out

    sections = [{"id": s["id"], "title": s["title"],
                 "blocks": _blocks(s["blocks"]),
                 "subsections": [{"id": sub["id"], "title": sub["title"],
                                  "blocks": _blocks(sub["blocks"])}
                                 for sub in s.get("subsections", ())]}
                for s in DOC_SECTIONS]
    return {"dataset": dataset, "entity": entity, "sections": sections}


def entity_summary(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """Compact card for the "previous results" list — no narrative loading."""
    global_ir = load_global_ir(dataset, entity)
    gtext = global_text_path(dataset, entity)
    if global_ir is None and gtext is None:
        return None
    faith = load_faithfulness(dataset, entity)
    decision = (global_ir or {}).get("decision") or {}
    return {
        "dataset": dataset,
        "dataset_label": dataset_label(dataset),
        "entity": entity,
        "framework_choice": decision.get("framework_choice"),
        "chosen": decision.get("chosen"),
        "n_stages": len([s for s in ((global_ir or {}).get("stages") or {}).values()
                         if s.get("status") == "ok"]),
        "hallucination_rate": ((faith or {}).get("overall") or {}).get("hallucination_rate"),
        "omission_rate": ((faith or {}).get("overall") or {}).get("omission_rate"),
        "generated_at": os.path.getmtime(gtext) if gtext else None,
    }


def known_entities() -> List[Tuple[str, str]]:
    """(dataset, entity) pairs that have explanation artifacts on disk."""
    found = []
    for root in (paths.EXPLANATIONS_NL, paths.EXPLANATIONS_IR):
        if not root.is_dir():
            continue
        for ds_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not ds_dir.is_dir():
                continue
            for ent_dir in sorted(ds_dir.iterdir(), key=lambda p: paths.natural_key(p.name)):
                if ent_dir.is_dir():
                    pair = (ds_dir.name, ent_dir.name)
                    if pair not in found:
                        found.append(pair)
    return found
