"""Calibration maths, checked against cases with known answers."""

from __future__ import annotations

from atlas_jev.evaluation.calibration import (
    precision_recall_curve,
    reliability_diagram,
    roc_curve,
)


class TestROC:
    def test_perfect_separation_gives_auc_one(self):
        assert roc_curve([(0.9, True), (0.8, True), (0.2, False), (0.1, False)])["auc"] == 1.0

    def test_inverted_ranking_gives_auc_zero(self):
        assert roc_curve([(0.1, True), (0.2, True), (0.8, False), (0.9, False)])["auc"] == 0.0

    def test_uninformative_scores_give_auc_half(self):
        """All ties. AUC 0.5 is the honest answer, and this is the result that
        would falsify the claim that the probability carries signal."""
        pairs = [(0.5, True), (0.5, False), (0.5, True), (0.5, False)]
        assert roc_curve(pairs)["auc"] == 0.5

    def test_auc_undefined_without_both_classes(self):
        result = roc_curve([(0.9, True), (0.8, True)])
        assert result["auc"] is None
        assert "undefined" in result["note"]

    def test_empty_input_is_safe(self):
        assert roc_curve([])["auc"] is None


class TestReliability:
    def test_well_calibrated_has_near_zero_ece(self):
        """8 of 10 positives at p=0.8 is perfectly calibrated."""
        pairs = [(0.8, True)] * 8 + [(0.8, False)] * 2
        result = reliability_diagram(pairs)
        assert result["ece"] < 1e-9
        assert abs(result["brier"] - 0.16) < 1e-9

    def test_overconfident_and_wrong_has_high_ece(self):
        result = reliability_diagram([(0.99, False)] * 10)
        assert result["ece"] > 0.95

    def test_discrimination_and_calibration_are_independent(self):
        """Perfect ranking can coexist with poor calibration. This is why the
        project reports AUC and ECE separately rather than one 'accuracy'."""
        pairs = [(0.55, True)] * 10 + [(0.45, False)] * 10
        result = reliability_diagram(pairs)
        assert roc_curve(pairs)["auc"] == 1.0
        assert result["ece"] > 0.4

    def test_probability_of_one_lands_in_the_last_bin(self):
        result = reliability_diagram([(1.0, True)], n_bins=10)
        assert sum(b["count"] for b in result["bins"]) == 1
        assert result["bins"][-1]["count"] == 1

    def test_empty_input_is_safe(self):
        result = reliability_diagram([])
        assert result["ece"] is None and result["n"] == 0


class TestPrecisionRecall:
    def test_recall_reaches_one_at_the_lowest_cutoff(self):
        pairs = [(0.9, True), (0.4, True), (0.3, False)]
        points = precision_recall_curve(pairs)["points"]
        assert max(p["recall"] for p in points) == 1.0

    def test_no_positives_is_reported_not_crashed(self):
        assert "note" in precision_recall_curve([(0.9, False)])
