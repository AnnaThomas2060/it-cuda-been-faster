"""Print a telemetry report: python -m src.analytics [log.jsonl] [--gpu T4]"""

import argparse
import json

from src.analytics.fake_gain import audit
from src.analytics.roofline import PEAK_BANDWIDTH_GBPS
from src.analytics.telemetry import DEFAULT_LOG_PATH, load_records, summarize

parser = argparse.ArgumentParser(description="Summarize telemetry and flag fake gains.")
parser.add_argument("path", nargs="?", default=str(DEFAULT_LOG_PATH))
parser.add_argument("--gpu", choices=sorted(PEAK_BANDWIDTH_GBPS), default=None)
args = parser.parse_args()

records = load_records(args.path)
print(json.dumps(summarize(records), indent=2))

flagged = audit(records, peak_gbps=PEAK_BANDWIDTH_GBPS.get(args.gpu))
print(f"\nFlagged records: {len(flagged)} of {len(records)}")
for record, flags in flagged:
    print(f"  {record.trajectory_id} {record.task_id} turn {record.turn_index}: {', '.join(flags)}")
