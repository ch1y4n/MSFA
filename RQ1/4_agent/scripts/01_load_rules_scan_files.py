from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULE_DIR = ROOT / "rules"
DEFAULT_TARGET_DIR = ROOT / "repo"
DEFAULT_OUT_DIR = ROOT / "scan_results"
# Optional fallback: point RG_PATH (or RIPGREP_PATH) at an rg executable when
# `rg` is not on PATH. Prefer installing ripgrep and relying on PATH.
DEFAULT_RG_CANDIDATES = [
    Path(p)
    for p in (os.environ.get("RG_PATH"), os.environ.get("RIPGREP_PATH"))
    if p
]

RG_EXCLUDES = [
    "!.git/**",
    "!.hg/**",
    "!.svn/**",
    "!node_modules/**",
    "!vendor/**",
    "!dist/**",
    "!build/**",
    "!target/**",
    "!.next/**",
    "!.nuxt/**",
    "!.turbo/**",
    "!.cache/**",
    "!coverage/**",
    "!__pycache__/**",
    "!.venv/**",
    "!venv/**",
    "!*.png",
    "!*.jpg",
    "!*.jpeg",
    "!*.gif",
    "!*.webp",
    "!*.ico",
    "!*.pdf",
    "!*.zip",
    "!*.gz",
    "!*.tgz",
    "!*.rar",
    "!*.7z",
    "!*.exe",
    "!*.dll",
    "!*.so",
    "!*.dylib",
    "!*.bin",
    "!*.onnx",
    "!*.pt",
    "!*.pth",
    "!*.safetensors",
    "!*.ckpt",
    "!*.mp4",
    "!*.mov",
    "!*.mp3",
    "!*.wav",
    "!*.woff",
    "!*.woff2",
    "!*.ttf",
    "!*.otf",
    "!*.jar",
    "!*.wasm",
    "!*.db",
    "!*.sqlite",
    "!*.sqlite3",
    "!*.parquet",
    "!*.npy",
    "!*.npz",
    "!*.pkl",
    "!*.pickle",
]


@dataclass(frozen=True)
class CompiledPattern:
    rule_file: str
    rule_id: str
    rule_name: str
    severity: str
    kind: str
    strength: str
    pattern: str
    regex: re.Pattern[str]
    priority: int
    capture_context_lines: int
    dedupe_same_rule_same_line: bool


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return data


def compile_one(pattern: str, match_type: str) -> re.Pattern[str]:
    if match_type == "literal":
        pattern = re.escape(pattern)
    return re.compile(pattern)


def load_rules(rule_dir: Path) -> list[CompiledPattern]:
    compiled: list[CompiledPattern] = []
    for rule_path in sorted(rule_dir.glob("*.yaml")):
        data = read_yaml(rule_path)
        settings = data.get("settings", {}).get("static_scan", {})
        match_kinds = settings.get("match_kinds", ["literal", "regex", "partial"])
        priority_order = settings.get("dedupe", {}).get(
            "priority", ["regex", "literal", "partial"]
        )
        priority_map = {kind: index for index, kind in enumerate(priority_order)}
        dedupe_same_line = bool(
            settings.get("dedupe", {}).get("same_rule_same_line", True)
        )
        context_lines = int(settings.get("capture_context_lines", 5))

        for rule in data.get("rules", []):
            static_patterns = rule.get("static_patterns", {})
            for kind in match_kinds:
                spec = static_patterns.get(kind)
                if not spec:
                    continue
                match_type = spec.get("match_type", "regex")
                strength = spec.get("strength", "")
                for pattern in spec.get("patterns", []):
                    compiled.append(
                        CompiledPattern(
                            rule_file=rule_path.name,
                            rule_id=str(rule.get("id", "")),
                            rule_name=str(rule.get("name", "")),
                            severity=str(rule.get("severity", "")),
                            kind=kind,
                            strength=strength,
                            pattern=str(pattern),
                            regex=compile_one(str(pattern), match_type),
                            priority=priority_map.get(kind, len(priority_map)),
                            capture_context_lines=context_lines,
                            dedupe_same_rule_same_line=dedupe_same_line,
                        )
                    )
    return compiled


def write_rg_pattern_file(path: Path, patterns: list[CompiledPattern]) -> int:
    unique_patterns = sorted({item.pattern for item in patterns})
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for pattern in unique_patterns:
            handle.write(pattern)
            handle.write("\n")
    return len(unique_patterns)


def resolve_rg_exe(value: str | None) -> str:
    if value:
        path = Path(value)
        if path.exists():
            return str(path)
        found = shutil.which(value)
        if found:
            return found
        raise FileNotFoundError(f"rg executable not found: {value}")

    found = shutil.which("rg")
    if found:
        return found

    for candidate in DEFAULT_RG_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "rg.exe not found. Install ripgrep or pass --rg-exe C:\\path\\to\\rg.exe"
    )


def run_rg(
    rg_exe: str,
    scan_dir: Path,
    pattern_file: Path,
    max_file_mb: float,
    timeout_sec: int,
):
    max_size = (
        f"{int(max_file_mb)}M"
        if float(max_file_mb).is_integer()
        else f"{int(max_file_mb * 1024)}K"
    )
    cmd = [
        rg_exe,
        "--json",
        "--pcre2",
        "--hidden",
        "--no-ignore",
        "--line-number",
        "--column",
        "--with-filename",
        "--max-filesize",
        max_size,
        "-f",
        str(pattern_file),
    ]
    for exclude in RG_EXCLUDES:
        cmd.extend(["-g", exclude])
    cmd.append(str(scan_dir))

    return subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
    )


def repo_name_for(path: Path, target_dir: Path) -> str:
    rel = path.relative_to(target_dir)
    return rel.parts[0] if len(rel.parts) > 1 else "."


def github_url_for(repo: str) -> str:
    if "__" not in repo:
        return ""
    owner, name = repo.split("__", 1)
    if not owner or not name:
        return ""
    return f"https://github.com/{owner}/{name}"


def iter_project_dirs(target_dir: Path) -> list[Path]:
    if (target_dir / ".git").is_dir():
        return [target_dir]
    dirs = [path for path in target_dir.iterdir() if path.is_dir()]
    if dirs:
        return sorted(dirs, key=lambda path: path.name.lower())
    return [target_dir]


def progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rest:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{rest:04.1f}s"


def read_lines_cached(path: Path, cache: dict[Path, list[str]]) -> list[str]:
    if path in cache:
        return cache[path]
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    cache[path] = text.splitlines()
    return cache[path]


def context_for(
    path: Path,
    line_number: int,
    radius: int,
    cache: dict[Path, list[str]],
) -> tuple[str, str]:
    lines = read_lines_cached(path, cache)
    index = line_number - 1
    before_start = max(0, index - radius)
    after_end = min(len(lines), index + radius + 1)
    before = "\n".join(lines[before_start:index])
    after = "\n".join(lines[index + 1 : after_end])
    return before, after


def patterns_for_line(
    line: str,
    patterns: list[CompiledPattern],
) -> list[tuple[int, int, CompiledPattern, re.Match[str]]]:
    candidates = []
    for compiled in patterns:
        for match in compiled.regex.finditer(line):
            candidates.append((compiled.priority, match.start(), compiled, match))

    best_by_rule: dict[str, tuple[int, int, CompiledPattern, re.Match[str]]] = {}
    for item in candidates:
        compiled = item[2]
        key = f"{compiled.rule_file}:{compiled.rule_id}"
        if not compiled.dedupe_same_rule_same_line:
            key = f"{key}:{item[1]}:{item[3].group(0)}"
        current = best_by_rule.get(key)
        if current is None or (item[0], item[1]) < (current[0], current[1]):
            best_by_rule[key] = item

    return sorted(
        best_by_rule.values(),
        key=lambda item: (item[0], item[1], item[2].rule_file, item[2].rule_id),
    )


def rows_from_rg(
    rg_stdout: str,
    target_dir: Path,
    patterns: list[CompiledPattern],
    repo_override: str | None = None,
) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    matched_lines: set[tuple[str, int]] = set()
    line_cache: dict[Path, list[str]] = {}

    for raw in rg_stdout.splitlines():
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue

        data = event["data"]
        path_text = data["path"]["text"]
        file_path = Path(path_text).resolve()
        line_number = int(data["line_number"])
        line = data["lines"]["text"].rstrip("\n\r")
        matched_lines.add((str(file_path), line_number))

        try:
            rel = file_path.relative_to(target_dir).as_posix()
        except ValueError:
            rel = file_path.as_posix()
        repo = repo_override or repo_name_for(file_path, target_dir)

        for _, _, compiled, match in patterns_for_line(line, patterns):
            before, after = context_for(
                file_path,
                line_number,
                compiled.capture_context_lines,
                line_cache,
            )
            rows.append(
                {
                    "repo": repo,
                    "url": github_url_for(repo),
                    "file": rel,
                    "line": str(line_number),
                    "column": str(match.start() + 1),
                    "match": match.group(0),
                    "rule_file": compiled.rule_file,
                    "rule_id": compiled.rule_id,
                    "rule_name": compiled.rule_name,
                    "kind": compiled.kind,
                    "strength": compiled.strength,
                    "severity": compiled.severity,
                    "pattern": compiled.pattern,
                    "context_before": before,
                    "context_line": line,
                    "context_after": after,
                }
            )

    return rows, len(matched_lines)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_hits(path: Path, rows: list[dict[str, str]]) -> None:
    write_csv(
        path,
        [
            "repo",
            "url",
            "file",
            "line",
            "column",
            "match",
            "rule_file",
            "rule_id",
            "rule_name",
            "kind",
            "strength",
            "severity",
            "pattern",
            "context_before",
            "context_line",
            "context_after",
        ],
        rows,
    )


def project_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_repo: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = by_repo.setdefault(
            row["repo"],
            {
                "repo": row["repo"],
                "url": row["url"],
                "hit_count": 0,
                "files": set(),
                "rules": Counter(),
                "rule_files": Counter(),
                "matches": Counter(),
            },
        )
        item["hit_count"] += 1
        item["files"].add(row["file"])
        item["rules"][row["rule_id"]] += 1
        item["rule_files"][row["rule_file"]] += 1
        item["matches"][row["match"]] += 1

    out = []
    for item in sorted(by_repo.values(), key=lambda x: (-x["hit_count"], x["repo"])):
        out.append(
            {
                "repo": item["repo"],
                "url": item["url"],
                "hit_count": str(item["hit_count"]),
                "matched_files": str(len(item["files"])),
                "rule_files": "; ".join(
                    f"{name}:{count}" for name, count in item["rule_files"].most_common()
                ),
                "top_rules": "; ".join(
                    f"{name}:{count}" for name, count in item["rules"].most_common(10)
                ),
                "top_matches": "; ".join(
                    f"{name}:{count}" for name, count in item["matches"].most_common()
                ),
            }
        )
    return out


def write_project_summary(path: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = project_summary(rows)
    write_csv(
        path,
        [
            "repo",
            "url",
            "hit_count",
            "matched_files",
            "rule_files",
            "top_rules",
            "top_matches",
        ],
        summary,
    )
    return summary


def write_manual_review_csv(path: Path, summary: list[dict[str, str]]) -> None:
    rows = []
    for row in summary:
        rows.append(
            {
                **row,
                "manual_status": "pending",
                "manual_category": "",
                "manual_note": "",
            }
        )
    write_csv(
        path,
        [
            "repo",
            "url",
            "hit_count",
            "matched_files",
            "rule_files",
            "top_rules",
            "top_matches",
            "manual_status",
            "manual_category",
            "manual_note",
        ],
        rows,
    )


def esc_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_manual_review_md(path: Path, summary: list[dict[str, str]]) -> None:
    lines = [
        "# 01 Manual Review",
        "",
        "| Repo | Hits | Files | Rule files | Top rules | Top matches |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in summary:
        repo = f"[{row['repo']}]({row['url']})" if row["url"] else row["repo"]
        lines.append(
            "| "
            + " | ".join(
                [
                    repo,
                    row["hit_count"],
                    row["matched_files"],
                    esc_md(row["rule_files"]),
                    esc_md(row["top_rules"]),
                    esc_md(row["top_matches"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scan_errors(path: Path, rows: list[dict[str, str]]) -> None:
    write_csv(path, ["repo", "path", "returncode", "seconds", "stderr"], rows)


def scan_project(
    project_dir: Path,
    target_dir: Path,
    rg_exe: str,
    pattern_file: Path,
    patterns: list[CompiledPattern],
    max_file_mb: float,
    timeout_sec: int,
) -> dict[str, Any]:
    repo = project_dir.name
    started = time.perf_counter()
    try:
        result = run_rg(rg_exe, project_dir, pattern_file, max_file_mb, timeout_sec)
    except subprocess.TimeoutExpired as exc:
        seconds = time.perf_counter() - started
        return {
            "repo": repo,
            "status": "timeout",
            "rows": [],
            "matched_lines": 0,
            "seconds": seconds,
            "error": {
                "repo": repo,
                "path": str(project_dir),
                "returncode": "timeout",
                "seconds": f"{seconds:.3f}",
                "stderr": f"rg timed out after {timeout_sec}s",
            },
        }
    except OSError as exc:
        seconds = time.perf_counter() - started
        return {
            "repo": repo,
            "status": "error",
            "rows": [],
            "matched_lines": 0,
            "seconds": seconds,
            "error": {
                "repo": repo,
                "path": str(project_dir),
                "returncode": "oserror",
                "seconds": f"{seconds:.3f}",
                "stderr": str(exc),
            },
        }
    seconds = time.perf_counter() - started

    if result.returncode in (0, 1):
        rows, matched_lines = rows_from_rg(result.stdout, target_dir, patterns, repo)
        return {
            "repo": repo,
            "status": "hit" if rows else "none",
            "rows": rows,
            "matched_lines": matched_lines,
            "seconds": seconds,
            "error": None,
        }

    return {
        "repo": repo,
        "status": "error",
        "rows": [],
        "matched_lines": 0,
        "seconds": seconds,
        "error": {
            "repo": repo,
            "path": str(project_dir),
            "returncode": str(result.returncode),
            "seconds": f"{seconds:.3f}",
            "stderr": result.stderr[-4000:],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 01: load static special-token rules and scan files with rg."
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULE_DIR)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-file-mb", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project-timeout-sec", type=int, default=120)
    parser.add_argument("--rg-exe", default=None, help="Path to rg.exe if it is not on PATH.")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    rule_dir = args.rules.resolve()
    target_dir = args.target.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rg_exe = resolve_rg_exe(args.rg_exe)

    patterns = load_rules(rule_dir)
    rg_pattern_file = out_dir / "01_rg_patterns.txt"
    unique_rg_patterns = write_rg_pattern_file(rg_pattern_file, patterns)
    project_dirs = iter_project_dirs(target_dir)
    print(f"rules_dir={rule_dir}", flush=True)
    print(f"target_dir={target_dir}", flush=True)
    print(f"projects={len(project_dirs)}", flush=True)
    print(f"compiled_patterns={len(patterns)}", flush=True)
    print(f"rg_patterns={unique_rg_patterns}", flush=True)
    print(f"rg_exe={rg_exe}", flush=True)
    print(f"workers={args.workers}", flush=True)
    print(f"project_timeout_sec={args.project_timeout_sec}", flush=True)

    all_rows: list[dict[str, str]] = []
    scan_errors: list[dict[str, str]] = []
    matched_lines = 0
    started = time.perf_counter()

    workers = max(1, args.workers)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                scan_project,
                project_dir,
                target_dir,
                rg_exe,
                rg_pattern_file,
                patterns,
                args.max_file_mb,
                args.project_timeout_sec,
            )
            for project_dir in project_dirs
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                completed += 1
                elapsed = time.perf_counter() - started
                scan_errors.append(
                    {
                        "repo": "(unknown)",
                        "path": "",
                        "returncode": "exception",
                        "seconds": "",
                        "stderr": repr(exc),
                    }
                )
                print(
                    f"{progress_bar(completed, len(project_dirs))} "
                    f"{completed}/{len(project_dirs)} error hits=0 "
                    f"time=0.0s elapsed={format_seconds(elapsed)} "
                    f"eta=unknown repo=(unknown)",
                    flush=True,
                )
                continue
            completed += 1
            rows = result["rows"]
            all_rows.extend(rows)
            matched_lines += int(result["matched_lines"])
            if result["error"]:
                scan_errors.append(result["error"])

            elapsed = time.perf_counter() - started
            avg = elapsed / completed
            eta = avg * (len(project_dirs) - completed)
            print(
                f"{progress_bar(completed, len(project_dirs))} "
                f"{completed}/{len(project_dirs)} "
                f"{result['status']} hits={len(rows)} "
                f"time={format_seconds(result['seconds'])} "
                f"elapsed={format_seconds(elapsed)} "
                f"eta={format_seconds(eta)} "
                f"repo={result['repo']}",
                flush=True,
            )

    all_rows.sort(
        key=lambda row: (
            row["repo"].lower(),
            row["file"].lower(),
            int(row["line"]),
            int(row["column"]),
            row["rule_file"],
            row["rule_id"],
        )
    )
    scan_errors.sort(key=lambda row: row["repo"].lower())

    hits_path = out_dir / "01_file_hits.csv"
    summary_path = out_dir / "01_project_summary.csv"
    manual_csv_path = out_dir / "01_manual_review.csv"
    manual_md_path = out_dir / "01_manual_review.md"
    errors_path = out_dir / "01_scan_errors.csv"

    write_hits(hits_path, all_rows)
    summary = write_project_summary(summary_path, all_rows)
    write_manual_review_csv(manual_csv_path, summary)
    write_manual_review_md(manual_md_path, summary)
    write_scan_errors(errors_path, scan_errors)

    elapsed = time.perf_counter() - started
    print(f"elapsed={format_seconds(elapsed)}")
    print(f"matched_lines={matched_lines}")
    print(f"hits={len(all_rows)}")
    print(f"matched_projects={len(summary)}")
    print(f"errors={len(scan_errors)}")
    print(f"wrote={hits_path}")
    print(f"wrote={summary_path}")
    print(f"wrote={manual_csv_path}")
    print(f"wrote={manual_md_path}")
    print(f"wrote={errors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
