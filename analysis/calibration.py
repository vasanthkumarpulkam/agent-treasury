"""Scores logged probability estimates against real outcomes.

The central question this answers: **does the model know anything the market doesn't?**

Brier score = mean((predicted_prob - actual_outcome)^2). Lower is better.
The test that matters is not "is the model's Brier score good" -- it's "is the model's
Brier score BETTER THAN THE MARKET'S". A market price is already a strong, liquid,
crowd-sourced forecast. Beating it is the entire game; matching it means you have no edge
and every trade you place is just paying fees for the privilege of taking risk.

Run:  python3 -m analysis.calibration
"""
import sys
import logging

logger = logging.getLogger("calibration")


def brier_score(predictions, outcomes) -> float:
    if not predictions:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)


def calibration_bins(predictions, outcomes, n_bins: int = 10):
    """Group predictions into probability bins and compare predicted vs actual frequency.
    A well-calibrated model that says '70%' should be right about 70% of the time."""
    bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        in_bin = [(p, o) for p, o in zip(predictions, outcomes) if lo <= p < hi or (i == n_bins - 1 and p == 1.0)]
        if not in_bin:
            bins.append({"range": f"{lo:.1f}-{hi:.1f}", "n": 0, "predicted": None, "actual": None})
            continue
        avg_pred = sum(p for p, _ in in_bin) / len(in_bin)
        avg_actual = sum(o for _, o in in_bin) / len(in_bin)
        bins.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "n": len(in_bin),
            "predicted": avg_pred,
            "actual": avg_actual,
        })
    return bins


def analyze(db) -> dict:
    resolved = db.resolved_estimates()
    counts = db.estimate_counts()

    if not resolved:
        return {"status": "no_data", "counts": counts}

    outcomes = [e["outcome"] for e in resolved]
    model_probs = [e["model_prob"] for e in resolved]
    market_probs = [e["implied_prob"] for e in resolved]

    model_brier = brier_score(model_probs, outcomes)
    market_brier = brier_score(market_probs, outcomes)

    # Same comparison restricted to the subset the agent actually traded on -- the model
    # can be mediocre overall but good specifically where it saw large edge (or, more
    # commonly and more expensively, the reverse).
    traded = [e for e in resolved if e["traded"]]
    traded_model_brier = brier_score([e["model_prob"] for e in traded], [e["outcome"] for e in traded]) if traded else float("nan")
    traded_market_brier = brier_score([e["implied_prob"] for e in traded], [e["outcome"] for e in traded]) if traded else float("nan")

    beats_market = model_brier < market_brier

    return {
        "status": "ok",
        "counts": counts,
        "n_resolved": len(resolved),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "beats_market": beats_market,
        "brier_improvement": market_brier - model_brier,
        "n_traded_resolved": len(traded),
        "traded_model_brier": traded_model_brier,
        "traded_market_brier": traded_market_brier,
        "bins": calibration_bins(model_probs, outcomes),
    }


def format_report(result: dict) -> str:
    if result["status"] == "no_data":
        c = result["counts"]
        return (
            "No resolved estimates yet.\n"
            f"  logged estimates: {c['total']}  (traded: {c['traded']}, resolved: {c['resolved']})\n"
            "Run the resolution checker after markets have had time to settle:\n"
            "  python3 -m analysis.resolve\n"
        )

    lines = []
    lines.append("=" * 64)
    lines.append("CALIBRATION REPORT")
    lines.append("=" * 64)
    c = result["counts"]
    lines.append(f"Estimates logged : {c['total']}   traded: {c['traded']}   resolved: {c['resolved']}")
    lines.append("")
    lines.append(f"Model  Brier score : {result['model_brier']:.4f}   (lower is better)")
    lines.append(f"Market Brier score : {result['market_brier']:.4f}   (the benchmark to beat)")
    lines.append(f"Improvement        : {result['brier_improvement']:+.4f}")
    lines.append("")

    n = result["n_resolved"]
    if result["beats_market"]:
        lines.append(f"VERDICT: model beat the market on {n} resolved markets.")
        if n < 100:
            lines.append(f"  ...but {n} samples is far too few to conclude anything. Noise looks")
            lines.append("  exactly like this. Keep collecting; ~200+ resolutions before believing it.")
    else:
        lines.append(f"VERDICT: model did NOT beat the market on {n} resolved markets.")
        lines.append("  On this evidence there is no edge, and trading it costs fees + spread.")
        lines.append("  Do not scale capital into this. Improve the estimator or don't trade it.")

    if result["n_traded_resolved"]:
        lines.append("")
        lines.append(f"On the {result['n_traded_resolved']} resolved markets it actually traded:")
        lines.append(f"  model Brier : {result['traded_model_brier']:.4f}")
        lines.append(f"  market Brier: {result['traded_market_brier']:.4f}")

    lines.append("")
    lines.append("Calibration (predicted vs actual frequency):")
    lines.append(f"  {'bin':<12}{'n':>6}{'predicted':>12}{'actual':>10}")
    for b in result["bins"]:
        if b["n"] == 0:
            continue
        lines.append(f"  {b['range']:<12}{b['n']:>6}{b['predicted']:>12.3f}{b['actual']:>10.3f}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main():
    import yaml
    from storage.db import Database

    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    db = Database(config["logging"]["db_path"])
    print(format_report(analyze(db)))


if __name__ == "__main__":
    main()
