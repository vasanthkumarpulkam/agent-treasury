"""Tests for the Brier scoring that decides whether the model has any edge at all."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from analysis.calibration import brier_score, calibration_bins
from pods.polymarket.resolution_checker import resolve_outcome


def test_brier_score_perfect_prediction():
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)


def test_brier_score_worst_prediction():
    assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)


def test_brier_score_coin_flip():
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_calibration_bins_group_correctly():
    preds = [0.05, 0.15, 0.95]
    outs = [0, 0, 1]
    bins = calibration_bins(preds, outs, n_bins=10)
    assert bins[0]["n"] == 1   # 0.0-0.1
    assert bins[1]["n"] == 1   # 0.1-0.2
    assert bins[9]["n"] == 1   # 0.9-1.0


def test_resolve_outcome_requires_decisive_settlement():
    assert resolve_outcome({"closed": True, "outcomePrices": '["1", "0"]'}) == 1
    assert resolve_outcome({"closed": True, "outcomePrices": '["0", "1"]'}) == 0
    # Closed but ambiguous (still disputed) must NOT be scored.
    assert resolve_outcome({"closed": True, "outcomePrices": '["0.5", "0.5"]'}) is None
    # Not closed yet.
    assert resolve_outcome({"closed": False, "outcomePrices": '["0.9", "0.1"]'}) is None
