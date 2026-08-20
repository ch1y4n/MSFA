"""Batch-clone the 875 audited agent projects into ``static/repo/``.

Reads ``agent_projects_875.csv`` (columns: name,url,source) and shallow-clones
each ``owner/repo`` into ``static/repo/<owner>__<repo>`` using the same naming
convention expected by ``01_load_rules_scan_files.py``. Existing directories are
skipped, so the script is resumable. A status log is written next to the list.

Usage:
    python scripts/00_clone_agent_repos.py                 # clone all, 8 workers
    python scripts/00_clone_agent_repos.py --workers 16
    python scripts/00_clone_agent_repos.py --list <csv> --out <dir>
    python scripts/00_clone_agent_repos.py --full           # full history (no --depth 1)
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIST = ROOT / "agent_projects_875.csv"
DEFAULT_OUT = ROOT / "repo"


def dir_name(full_name: str) -> str:
    """``owner/repo`` -> ``owner__repo`` (matches the scanner's layout)."""
    return full_name.replace("/", "__")


def load_list(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clone_one(row: dict, out_dir: Path, depth_args: list[str], timeout: int) -> dict:
    name = (row.get("name") or "").strip()
    url = (row.get("url") or f"https://github.com/{name}").strip()
    if not name:
        return {"name": name, "url": url, "status": "skip_empty", "seconds": 0, "message": ""}

    target = out_dir / dir_name(name)
    if target.exists():
        return {"name": name, "url": url, "status": "skip_existing", "seconds": 0, "message": ""}

    tmp = out_dir / (dir_name(name) + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    start = time.time()
    cmd = ["git", "clone", *depth_args, url + ".git", str(tmp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"name": name, "url": url, "status": "timeout", "seconds": round(time.time() - start, 1), "message": f"timeout>{timeout}s"}
    secs = round(time.time() - start, 1)
    if proc.returncode == 0:
        tmp.rename(target)
        return {"name": name, "url": url, "status": "cloned", "seconds": secs, "message": ""}
    shutil.rmtree(tmp, ignore_errors=True)
    msg = (proc.stderr or proc.stdout or "").strip().splitlines()
    return {"name": name, "url": url, "status": "error", "seconds": secs, "message": (msg[-1] if msg else "")[:300]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-clone audited agent repos for static scanning.")
    ap.add_argument("--list", type=Path, default=DEFAULT_LIST, help="CSV with name/url columns")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output repo directory")
    ap.add_argument("--workers", type=int, default=8, help="parallel git clones")
    ap.add_argument("--timeout", type=int, default=600, help="per-repo clone timeout (seconds)")
    ap.add_argument("--full", action="store_true", help="clone full history (default is --depth 1)")
    args = ap.parse_args()

    if shutil.which("git") is None:
        print("ERROR: git not found on PATH", file=sys.stderr)
        return 2
    if not args.list.exists():
        print(f"ERROR: list not found: {args.list}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    depth_args = [] if args.full else ["--depth", "1", "--single-branch"]
    rows = load_list(args.list)
    total = len(rows)
    print(f"[+] {total} repos -> {args.out}  (workers={args.workers}, "
          f"{'full' if args.full else 'shallow'} clone)")

    results: list[dict] = []
    counts = {"cloned": 0, "skip_existing": 0, "error": 0, "timeout": 0, "skip_empty": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(clone_one, r, args.out, depth_args, args.timeout): r for r in rows}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            done += 1
            flag = {"cloned": "OK", "skip_existing": "==", "error": "!!", "timeout": "TO", "skip_empty": ".."}.get(res["status"], "??")
            print(f"[{done}/{total}] {flag} {res['name']}"
                  + (f"  ({res['message']})" if res["message"] else ""))

    log_path = args.list.with_name("clone_status.csv")
    with log_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "url", "status", "seconds", "message"])
        w.writeheader()
        w.writerows(results)

    print(f"\n[done] " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    print(f"[done] status log -> {log_path}")
    print(f"[next] scan with: python scripts/01_load_rules_scan_files.py")
    return 1 if (counts.get("error") or counts.get("timeout")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
