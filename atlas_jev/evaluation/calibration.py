"""Confidence and threshold analysis - the central experiment.

Why this module is the point of the project
-------------------------------------------
Jev returns a ``Noul`` answer as a bare probability, p(yes), with no confidence
field. A boolean therefore requires a threshold, and the threshold is a free
parameter. The interesting question is not "did Jev detect the jailbreak" but:

    Does Jev's probability carry enough information to control the
    false-positive / false-negative trade-off of an automated security decision?

That question has a precise answer shape, and this module computes it:

* a **threshold sweep** - precision, recall, FPR, FNR at each cutoff,
* an **ROC / PR curve** with AUC, which is threshold-independent,
* a **reliability diagram** with **ECE** and **Brier score**, which ask whether
  the number behaves like a probability at all.

The distinction between those last two matters. A probability can rank cases
perfectly (AUC 1.0) while being badly calibrated (ECE high), and vice versa.
Discrimination and calibration are separate properties, so H2 splits:

**H2a** - ``noul`` p(yes) is calibrated: when Jev says 0.8, it is right about
80% of the time. Tested by ECE and the reliability diagram.

**H2b** - ``Choice``/``Score`` ``confidence`` predicts correctness. Note that
the vendor documents this value as ``(n * peak - 1) / (n - 1)``, a deterministic
function of the returned distribution. It is a *dispersion statistic*, not a
validated probability of being correct, so whether it predicts correctness is an
open empirical question rather than an assumption.

Every figure here is computed from stored probabilities, so the full sweep costs
zero additional API calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from atlas_jev.evaluation.metrics import (
    Confusion,
    answer_is_correct,
    confusion_for,
    ground_truth_for,
    probability_for,
)

DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass
class ThresholdPoint:
    threshold: float
    confusion: Confusion

    def to_json(self) -> dict[str, Any]:
        return {"threshold": self.threshold, **self.confusion.to_json()}


def threshold_sweep(
    records: Sequence[dict[str, Any]],
    evaluator: str,
    question: str = "attack_success",
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    label_store: Any = None,
) -> list[ThresholdPoint]:
    """Confusion matrix at each decision threshold, from stored probabilities."""
    return [
        ThresholdPoint(
            threshold=t,
            confusion=confusion_for(records, evaluator, question, t, label_store),
        )
        for t in thresholds
    ]


def _labelled_pairs(
    records: Sequence[dict[str, Any]],
    evaluator: str,
    question: str,
    label_store: Any = None,
) -> list[tuple[float, bool]]:
    """``(probability, ground_truth)`` for every record that has both."""
    pairs: list[tuple[float, bool]] = []
    for record in records:
        if record.get("error"):
            continue
        truth = ground_truth_for(record, question, label_store)
        probability = probability_for(record, evaluator, question)
        if truth is None or probability is None:
            continue
        pairs.append((probability, truth))
    return pairs


# ---------------------------------------------------------------------------
# Discrimination: can the probability rank cases correctly?
# ---------------------------------------------------------------------------
def roc_curve(pairs: Sequence[tuple[float, bool]]) -> dict[str, Any]:
    """ROC points and AUC, computed exactly (no sklearn dependency).

    AUC is the Mann-Whitney U statistic: the probability that a randomly chosen
    positive is scored above a randomly chosen negative, with ties counting 0.5.
    """
    positives = [p for p, y in pairs if y]
    negatives = [p for p, y in pairs if not y]
    if not positives or not negatives:
        return {
            "auc": None,
            "points": [],
            "n_positive": len(positives),
            "n_negative": len(negatives),
            "note": "AUC undefined - need at least one positive and one negative label",
        }

    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    )
    auc = wins / (len(positives) * len(negatives))

    points: list[dict[str, float]] = []
    for cutoff in sorted({p for p, _ in pairs} | {0.0, 1.0}, reverse=True):
        tp = sum(1 for p in positives if p >= cutoff)
        fp = sum(1 for p in negatives if p >= cutoff)
        points.append(
            {
                "threshold": cutoff,
                "tpr": tp / len(positives),
                "fpr": fp / len(negatives),
            }
        )

    return {
        "auc": auc,
        "points": points,
        "n_positive": len(positives),
        "n_negative": len(negatives),
    }


def precision_recall_curve(pairs: Sequence[tuple[float, bool]]) -> dict[str, Any]:
    positives = [p for p, y in pairs if y]
    if not positives:
        return {"points": [], "note": "no positive labels"}
    points: list[dict[str, float]] = []
    for cutoff in sorted({p for p, _ in pairs}, reverse=True):
        predicted = [(p, y) for p, y in pairs if p >= cutoff]
        tp = sum(1 for _, y in predicted if y)
        precision = tp / len(predicted) if predicted else 0.0
        points.append(
            {
                "threshold": cutoff,
                "precision": precision,
                "recall": tp / len(positives),
            }
        )
    return {"points": points, "n_positive": len(positives)}


# ---------------------------------------------------------------------------
# Calibration: does the number behave like a probability?
# ---------------------------------------------------------------------------
def reliability_diagram(
    pairs: Sequence[tuple[float, bool]], n_bins: int = 10
) -> dict[str, Any]:
    """Binned observed frequency vs predicted probability, with ECE and Brier.

    ECE (expected calibration error) is the sample-weighted mean gap between a
    bin's mean predicted probability and its observed positive rate. Brier score
    is the mean squared error of the probability itself.

    A caution that belongs in every reading of this output: with a few dozen
    labels, per-bin counts are tiny and ECE is a noisy estimate. It indicates,
    it does not establish. That is a limitation of the sample size, not
    something a bigger number of bins can fix.
    """
    if not pairs:
        return {"bins": [], "ece": None, "brier": None, "n": 0}

    bins: list[dict[str, Any]] = []
    total = len(pairs)
    ece = 0.0
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        # Last bin is closed on the right so p == 1.0 is included.
        members = [
            (p, y)
            for p, y in pairs
            if (low <= p < high) or (index == n_bins - 1 and p == 1.0)
        ]
        if not members:
            bins.append(
                {
                    "bin_low": low,
                    "bin_high": high,
                    "count": 0,
                    "mean_predicted": None,
                    "observed_rate": None,
                    "gap": None,
                }
            )
            continue
        mean_predicted = sum(p for p, _ in members) / len(members)
        observed = sum(1 for _, y in members if y) / len(members)
        gap = abs(mean_predicted - observed)
        ece += (len(members) / total) * gap
        bins.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": len(members),
                "mean_predicted": mean_predicted,
                "observed_rate": observed,
                "gap": gap,
            }
        )

    brier = sum((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs) / total
    return {"bins": bins, "ece": ece, "brier": brier, "n": total}


def confidence_vs_correctness(
    records: Sequence[dict[str, Any]],
    evaluator: str,
    question: str,
    label_store: Any = None,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Does higher confidence actually mean more often correct?

    Works for any question kind, using whichever confidence value the answer
    carries, and reports ``confidence_source`` so an API-supplied dispersion
    statistic is never silently compared against our own derived certainty.

    Also reports the AUC of confidence as a predictor of correctness. An AUC
    near 0.5 means confidence carries no information about correctness - a
    result worth reporting plainly if that is what the data shows.
    """
    observations: list[tuple[float, bool]] = []
    sources: set[str] = set()
    for record in records:
        if record.get("error"):
            continue
        # Works for bool, choice and score alike. Restricting this to bool was
        # what kept the API's own Choice/Score confidence out of the analysis,
        # leaving only our derived 2|p-0.5| - which is a function of the same
        # probability that produced the decision, and so nearly tautological.
        correct = answer_is_correct(record, evaluator, question, label_store)
        if correct is None:
            continue
        answer = record["evaluations"][evaluator]["answers"][question]
        confidence = answer.get("confidence")
        if confidence is None:
            continue
        sources.add(answer.get("confidence_source", "unknown"))
        observations.append((confidence, correct))

    if not observations:
        return {
            "bins": [],
            "n": 0,
            "auc_confidence_predicts_correct": None,
            "confidence_sources": sorted(sources),
            "note": "no records had both a confidence value and a ground-truth label",
        }

    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        members = [
            (c, ok)
            for c, ok in observations
            if (low <= c < high) or (index == n_bins - 1 and c == 1.0)
        ]
        bins.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": len(members),
                "mean_confidence": (
                    sum(c for c, _ in members) / len(members) if members else None
                ),
                "accuracy": (
                    sum(1 for _, ok in members if ok) / len(members) if members else None
                ),
            }
        )

    correct = [c for c, ok in observations if ok]
    wrong = [c for c, ok in observations if not ok]
    if correct and wrong:
        wins = sum(
            1.0 if c > w else 0.5 if c == w else 0.0 for c in correct for w in wrong
        )
        auc = wins / (len(correct) * len(wrong))
    else:
        auc = None

    return {
        "bins": bins,
        "n": len(observations),
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "auc_confidence_predicts_correct": auc,
        "confidence_sources": sorted(sources),
        "overall_accuracy": len(correct) / len(observations),
    }


def calibration_report(
    records: Sequence[dict[str, Any]],
    evaluator: str,
    question: str = "attack_success",
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    label_store: Any = None,
) -> dict[str, Any]:
    """Everything needed for the confidence analysis and figures 3 and 4."""
    pairs = _labelled_pairs(records, evaluator, question, label_store)
    return {
        "evaluator": evaluator,
        "question": question,
        "n_labelled": len(pairs),
        "threshold_sweep": [p.to_json() for p in threshold_sweep(
            records, evaluator, question, thresholds, label_store
        )],
        "roc": roc_curve(pairs),
        "precision_recall": precision_recall_curve(pairs),
        "reliability": reliability_diagram(pairs),
        "confidence_vs_correctness": confidence_vs_correctness(
            records, evaluator, question, label_store
        ),
        # H2b proper: the API-supplied confidence on Choice/Score answers,
        # scored against correctness for every question that has ground truth.
        "confidence_by_question": {
            name: confidence_vs_correctness(records, evaluator, name, label_store)
            for name in _questions_present(records, evaluator)
        },
    }


def _questions_present(records: Sequence[dict[str, Any]], evaluator: str) -> list[str]:
    names: list[str] = []
    for record in records:
        answers = (record.get("evaluations", {}).get(evaluator) or {}).get("answers", {})
        for name in answers:
            if name not in names:
                names.append(name)
    return names
