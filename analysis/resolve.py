"""Check unresolved estimates against Polymarket and record outcomes.

Run periodically (daily is plenty):  python3 -m analysis.resolve
"""
import logging
import yaml

from storage.db import Database
from pods.polymarket.resolution_checker import check_and_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    db = Database(config["logging"]["db_path"])
    stats = check_and_record(db)
    print(f"checked={stats['checked']} resolved={stats['resolved']} errors={stats['errors']}")
    print(f"totals: {db.estimate_counts()}")


if __name__ == "__main__":
    main()
