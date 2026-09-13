"""Show what the models actually predicted, and why trades were or weren't taken.

When the agent reports "0 proposals from 15 markets", that is either the risk gates doing
their job or the gates being mistuned -- and those look identical from the outside. This
shows the underlying estimates so you can tell the difference.

Run:  python3 -m analysis.estimates [--limit 30] [--traded-only]
"""
import argparse
import yaml

from storage.db import Database


def load_recent(db, limit: int, traded_only: bool):
    with db._conn() as conn:
        sql = "SELECT * FROM estimates"
        if traded_only:
            sql += " WHERE traded=1"
        sql += " ORDER BY ts DESC LIMIT ?"
        return [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]


def classify(row, cfg):
    """Why didn't this become a trade? Mirrors the gates in research_agent."""
    if row["llm_confidence"] is None or row["llm_confidence"] <= 0:
        return "no estimate (LLM failed / no key)"
    edge = row["edge"] or 0.0
    implied = row["implied_prob"]
    model_p = row["model_prob"]

    if abs(edge) < cfg.get("min_edge_pct", 0.06):
        return f"edge {edge:+.3f} below min {cfg.get('min_edge_pct', 0.06)}"

    lo, hi = cfg.get("min_price", 0.05), cfg.get("max_price", 0.95)
    if not (lo <= implied <= hi):
        return f"price {implied:.4f} outside band [{lo}, {hi}]"

    max_ratio = cfg.get("max_odds_ratio", 3.0)
    if implied > 0 and model_p > 0:
        ratio = max(model_p / implied, implied / model_p)
        if ratio > max_ratio:
            return f"{ratio:.1f}x claim exceeds max_odds_ratio {max_ratio}"

    return "TRADED" if row["traded"] else "passed gates"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--traded-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(open("config/settings.yaml"))
    db = Database(config["logging"]["db_path"])
    rows = load_recent(db, args.limit, args.traded_only)

    if not rows:
        print("No estimates logged yet. Run: python3 main.py")
        return

    counts = db.estimate_counts()
    print("=" * 100)
    print(f"RECENT ESTIMATES  (showing {len(rows)} of {counts['total']} logged, "
          f"{counts['traded']} traded, {counts['resolved']} resolved)")
    print("=" * 100)

    reasons = {}
    for r in rows:
        verdict = classify(r, config["polymarket"])
        reasons[verdict.split(" ")[0]] = reasons.get(verdict.split(" ")[0], 0) + 1

        question = (r["question"] or "")[:58]
        print(f"\n{question}")
        print(f"  market {r['implied_prob']:.4f}  ->  model {r['model_prob']:.4f}   "
              f"edge {(r['edge'] or 0):+.3f}   conf {(r['llm_confidence'] or 0):.2f}")
        print(f"  model: {r['model_name']}")
        print(f"  verdict: {verdict}")
        if r["reasoning"]:
            print(f"  reasoning: {(r['reasoning'] or '')[:150]}")
        if r["outcome"] is not None:
            hit = (r["model_prob"] > 0.5) == (r["outcome"] == 1)
            print(f"  RESOLVED: {'YES' if r['outcome'] else 'NO'} -- model was {'right' if hit else 'WRONG'}")

    print("\n" + "=" * 100)
    print("WHY NO TRADE (this sample):")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {reason}")
    print("=" * 100)
    print("If almost everything is 'edge below min', the models broadly agree with the")
    print("market -- which is the expected and honest result, not a bug to tune away.")
    print("Loosening thresholds to force trades manufactures activity, not edge.")


if __name__ == "__main__":
    main()
