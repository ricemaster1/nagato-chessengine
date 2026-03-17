#!/usr/bin/env python3
"""Summarize Nagato static Elo reference from benchmark artifacts.

Reads:
- training/elo_reference_history.jsonl
- training/elo_reference_latest.json
- training/elo_reference_config.json

Prints:
- one-line status for the latest reliable finite estimate
- latest run context (even if non-finite)
- short trend across recent finite runs
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass
class RunSummary:
    timestamp: str | None
    candidate: str | None
    reference: str | None
    reference_elo: int | None
    tc: str | None
    games: int | None
    wins: int | None
    draws: int | None
    losses: int | None
    delta_elo: float | None
    delta_ci_lo: float | None
    delta_ci_hi: float | None
    abs_elo: float | None

    @property
    def finite(self) -> bool:
        return self.abs_elo is not None and math.isfinite(self.abs_elo)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report Nagato static Elo reference")
    parser.add_argument("--config", default="training/elo_reference_config.json")
    parser.add_argument("--latest", default="training/elo_reference_latest.json")
    parser.add_argument("--history", default="training/elo_reference_history.jsonl")
    parser.add_argument("--trend", type=int, default=5, help="finite runs to include in trend")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON report")
    return parser.parse_args()


def safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def safe_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def from_record(rec: dict) -> RunSummary:
    # Newest script writes this in config.
    if "last_run" in rec and isinstance(rec["last_run"], dict):
        lr = rec["last_run"]
        wdl = lr.get("wdl", {})
        ci = lr.get("elo_vs_reference_95ci", [None, None])
        return RunSummary(
            timestamp=lr.get("timestamp_utc"),
            candidate=lr.get("candidate"),
            reference=lr.get("reference"),
            reference_elo=safe_int(lr.get("reference_uci_elo")),
            tc=lr.get("tc"),
            games=safe_int(lr.get("games")),
            wins=safe_int(wdl.get("wins")),
            draws=safe_int(wdl.get("draws")),
            losses=safe_int(wdl.get("losses")),
            delta_elo=safe_float(lr.get("elo_vs_reference")),
            delta_ci_lo=safe_float(ci[0] if len(ci) > 0 else None),
            delta_ci_hi=safe_float(ci[1] if len(ci) > 1 else None),
            abs_elo=safe_float(lr.get("estimated_absolute_elo")),
        )

    # History/latest record format from elo_reference.py.
    match = rec.get("match", {})
    engines = rec.get("engines", {})
    ref_opts = engines.get("reference", {}).get("options", {})
    ref_elo = safe_int(ref_opts.get("UCI_Elo"))
    delta = safe_float(match.get("elo_vs_opponent"))
    ci = match.get("elo_95ci", [None, None])
    abs_elo = None
    if ref_elo is not None and delta is not None:
        abs_elo = ref_elo + delta

    return RunSummary(
        timestamp=rec.get("timestamp_utc"),
        candidate=engines.get("candidate", {}).get("name"),
        reference=engines.get("reference", {}).get("name"),
        reference_elo=ref_elo,
        tc=rec.get("protocol", {}).get("tc"),
        games=safe_int(match.get("games")),
        wins=safe_int(match.get("wins")),
        draws=safe_int(match.get("draws")),
        losses=safe_int(match.get("losses")),
        delta_elo=delta,
        delta_ci_lo=safe_float(ci[0] if len(ci) > 0 else None),
        delta_ci_hi=safe_float(ci[1] if len(ci) > 1 else None),
        abs_elo=abs_elo,
    )


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_history(path: Path) -> list[RunSummary]:
    if not path.exists():
        return []

    runs: list[RunSummary] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            runs.append(from_record(rec))
    return runs


def choose_latest_run(config_path: Path, latest_path: Path, history_runs: list[RunSummary]) -> RunSummary | None:
    latest = read_json(latest_path)
    if latest:
        return from_record(latest)

    config = read_json(config_path)
    if config:
        rs = from_record(config)
        if rs.timestamp or rs.games:
            return rs

    if history_runs:
        return history_runs[-1]
    return None


def format_run_line(prefix: str, run: RunSummary) -> str:
    w = run.wins if run.wins is not None else "?"
    d = run.draws if run.draws is not None else "?"
    l = run.losses if run.losses is not None else "?"
    g = run.games if run.games is not None else "?"
    tc = run.tc or "?"
    ref = run.reference or "reference"
    ref_elo = f"@{run.reference_elo}" if run.reference_elo is not None else ""

    if run.abs_elo is not None:
        return (
            f"{prefix} ~{run.abs_elo:.1f} Elo "
            f"(delta {run.delta_elo:+.1f} vs {ref}{ref_elo}, "
            f"WDL {w}/{d}/{l}, {g} games, tc {tc})"
        )

    return f"{prefix} unavailable (non-finite estimate); latest WDL {w}/{d}/{l}, {g} games, tc {tc}"


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    config_path = root / args.config
    latest_path = root / args.latest
    history_path = root / args.history

    history_runs = read_history(history_path)
    latest_run = choose_latest_run(config_path, latest_path, history_runs)
    finite_runs = [r for r in history_runs if r.finite]
    latest_finite = finite_runs[-1] if finite_runs else None

    trend_window = finite_runs[-args.trend :] if args.trend > 0 else []
    trend_abs_values = [r.abs_elo for r in trend_window if r.abs_elo is not None]
    trend_delta_values = [r.delta_elo for r in trend_window if r.delta_elo is not None]
    trend_abs_mean = mean(trend_abs_values) if trend_abs_values else None
    trend_delta_mean = mean(trend_delta_values) if trend_delta_values else None
    trend_drift = None
    if len(trend_abs_values) >= 2:
        trend_drift = trend_abs_values[-1] - trend_abs_values[0]

    report = {
        "latest": None if latest_run is None else latest_run.__dict__,
        "latest_finite": None if latest_finite is None else latest_finite.__dict__,
        "history_count": len(history_runs),
        "finite_history_count": len(finite_runs),
        "trend_window": len(trend_window),
        "trend_abs_mean": trend_abs_mean,
        "trend_delta_mean": trend_delta_mean,
        "trend_drift": trend_drift,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    if latest_finite:
        print(format_run_line("Static Elo reference:", latest_finite))
    elif latest_run:
        print(format_run_line("Static Elo reference:", latest_run))
    else:
        print("Static Elo reference: no benchmark data found yet")
        print(f"Run: python {args.config.replace('_config.json', '.py')}")
        return 0

    if latest_run and latest_finite and latest_run.timestamp != latest_finite.timestamp:
        print("Latest run note: most recent run is non-finite; using latest finite estimate from history.")
        print(format_run_line("Latest run:", latest_run))

    if trend_window:
        drift_txt = "n/a" if trend_drift is None else f"{trend_drift:+.1f}"
        delta_txt = "n/a" if trend_delta_mean is None else f"{trend_delta_mean:+.1f}"
        abs_txt = "n/a" if trend_abs_mean is None else f"{trend_abs_mean:.1f}"
        print(
            f"Trend ({len(trend_window)} finite runs): "
            f"mean abs {abs_txt}, mean delta {delta_txt}, drift {drift_txt} Elo"
        )

    print(f"Artifacts: {args.latest}, {args.history}, {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
