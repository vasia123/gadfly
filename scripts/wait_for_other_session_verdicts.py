"""Wait until any session OTHER than `MY_SESSION` has produced N+ verdicts.

Reads the gadfly log directory, ignores the named session file, and reports
when the rest have accumulated >= --min verdicts. Exits 0 on success, prints
a summary of the other-session verdicts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def count_other_verdicts(log_dir: Path, my_session: str) -> tuple[int, list[Path]]:
    count = 0
    files: list[Path] = []
    for f in log_dir.glob("*.jsonl"):
        if f.name == my_session:
            continue
        files.append(f)
        try:
            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return count, files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=str(Path.home() / ".claude" / "gadfly" / "log"))
    ap.add_argument("--my-session", required=True, help="Filename to exclude (mine).")
    ap.add_argument("--min", type=int, default=3)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--max-wait", type=float, default=3600.0)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + args.max_wait
    last_reported = -1
    while time.time() < deadline:
        n, files = count_other_verdicts(log_dir, args.my_session)
        if n != last_reported:
            print(f"[{time.strftime('%H:%M:%S')}] other-session verdicts: {n}/{args.min}"
                  f" across {len(files)} file(s)", flush=True)
            last_reported = n
        if n >= args.min:
            print("=== threshold reached ===", flush=True)
            for f in files:
                print(f"  {f}")
            return 0
        time.sleep(args.poll)

    print("=== timed out ===", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
