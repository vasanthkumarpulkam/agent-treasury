"""Start a new generation after death.

Deliberately a separate, explicit command a human must run with an explicit amount --
the system can never bring itself back to life. That asymmetry is the point: it can end
itself automatically, but only you can decide to risk more money on it.

Run:  python3 -m analysis.resurrect --capital 500 --note "tightened risk limits"
"""
import argparse
import logging
import yaml

from storage.db import Database
from governor.governor import Governor
from analysis.status import build_status, format_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, required=True, help="capital for the new generation")
    parser.add_argument("--note", default="", help="what you changed before trying again")
    args = parser.parse_args()

    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    db = Database(config["logging"]["db_path"])
    governor = Governor(db, config, execution_clients={})

    if not db.get_state("killed", False):
        print("This generation is still alive. Nothing to resurrect.")
        print(format_status(build_status(db, config)))
        return

    graves = db.graveyard()
    print(format_status(build_status(db, config)))
    print()
    if graves:
        total = sum(g["pnl"] for g in graves)
        print(f"You have already lost ${abs(total):,.2f} across {len(graves)} generation(s).")
    confirm = input(f"Start generation {db.get_state('generation', 1) + 1} with ${args.capital:,.2f}? [y/N] ")
    if confirm.strip().lower() != "y":
        print("Cancelled.")
        return

    governor.resurrect(args.capital, args.note or "no note given")
    print(format_status(build_status(db, config)))


if __name__ == "__main__":
    main()
