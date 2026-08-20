#!/usr/bin/env python3
"""Recompute the paper tables from the selected AgentDoJo result JSON files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


SUITES = ("travel", "banking", "slack", "workspace")
EXPECTED_COUNTS = {"travel": 140, "banking": 144, "slack": 105, "workspace": 560}
RQ2_ATTACKS = {
    "glm_sti": "STI",
    "glm_sti_vector": "SVS-STI",
    "glm_plain": "PI",
}
RQ3_ATTACKS = {
    "direct": "DI",
    "ignore_previous": "IP",
    "important_instructions": "II",
    "tool_knowledge": "TK",
    "sti_attack_promptarmor": "MSFA",
    "sti_attack_melon": "MSFA",
    "sti_attack_attriguard": "MSFA",
}
DEFENSE_NAMES = {
    "none": "No Defense",
    "promptarmor": "PromptArmor",
    "melon": "MELON",
    "attriguard": "AttriGuard",
}


def defense_from_pipeline(pipeline: str) -> str:
    for name in ("promptarmor", "melon", "attriguard"):
        if pipeline.endswith("-" + name):
            return name
    return "none"


def load_records(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        attack = row.get("attack_type")
        if not attack or row.get("error") is not None:
            continue
        row["_path"] = str(path)
        records.append(row)
    return records


def rates(rows: list[dict]) -> tuple[float, float]:
    n = len(rows)
    if not n:
        raise ValueError("empty result group")
    asr = 100.0 * sum(row.get("security") is True for row in rows) / n
    utility = 100.0 * sum(row.get("utility") is True for row in rows) / n
    return asr, utility


def cell(rows: list[dict]) -> str:
    asr, utility = rates(rows)
    return f"{asr:.1f} / {utility:.1f}"


def summarize_rq2(records: list[dict]) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        attack = row.get("attack_type")
        suite = row.get("suite_name")
        if attack in RQ2_ATTACKS and suite in SUITES:
            groups[(attack, suite)].append(row)

    expected = {(attack, suite) for attack in RQ2_ATTACKS for suite in SUITES}
    if set(groups) != expected:
        raise SystemExit(f"RQ2 groups differ from expected: {sorted(set(groups) ^ expected)}")

    print("| Attack | Travel | Banking | Slack | Workspace |")
    print("|---|---:|---:|---:|---:|")
    for attack in ("glm_sti", "glm_sti_vector", "glm_plain"):
        cells = []
        for suite in SUITES:
            rows = groups[(attack, suite)]
            if len(rows) != EXPECTED_COUNTS[suite]:
                raise SystemExit(f"unexpected count for {attack}/{suite}: {len(rows)}")
            cells.append(cell(rows))
        print(f"| {RQ2_ATTACKS[attack]} | " + " | ".join(cells) + " |")


def summarize_rq3(records: list[dict]) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        attack = row.get("attack_type")
        suite = row.get("suite_name")
        if attack not in RQ3_ATTACKS or suite not in SUITES:
            continue
        defense = defense_from_pipeline(str(row.get("pipeline_name", "")))
        groups[(defense, RQ3_ATTACKS[attack], suite)].append(row)

    expected = {
        (defense, attack, suite)
        for defense in DEFENSE_NAMES
        for attack in (("DI", "IP", "II", "TK") if defense == "none" else ("DI", "IP", "II", "TK", "MSFA"))
        for suite in SUITES
    }
    if set(groups) != expected:
        raise SystemExit(f"RQ3 groups differ from expected: {sorted(set(groups) ^ expected)}")

    print("| Defense | Attack | Travel | Banking | Slack | Workspace | Average |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for defense in ("none", "promptarmor", "melon", "attriguard"):
        attacks = ("DI", "IP", "II", "TK") if defense == "none" else ("DI", "IP", "II", "TK", "MSFA")
        for attack in attacks:
            suite_rates = []
            cells = []
            for suite in SUITES:
                rows = groups[(defense, attack, suite)]
                if len(rows) != EXPECTED_COUNTS[suite]:
                    raise SystemExit(f"unexpected count for {defense}/{attack}/{suite}: {len(rows)}")
                asr, utility = rates(rows)
                suite_rates.append((round(asr, 1), round(utility, 1)))
                cells.append(f"{asr:.1f} / {utility:.1f}")
            avg_asr = mean(value[0] for value in suite_rates)
            avg_utility = mean(value[1] for value in suite_rates)
            avg = f"{avg_asr:.1f} / {avg_utility:.1f}"
            print(f"| {DEFENSE_NAMES[defense]} | {attack} | " + " | ".join(cells) + f" | {avg} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq", choices=("2", "3"), required=True)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "runs")
    args = parser.parse_args()
    records = load_records(args.results)
    if args.rq == "2":
        summarize_rq2(records)
    else:
        summarize_rq3(records)


if __name__ == "__main__":
    main()
