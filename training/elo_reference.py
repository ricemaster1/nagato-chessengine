#!/usr/bin/env python3
"""Run a reproducible Elo reference match and persist results.

This script uses cutechess-cli with a fixed protocol to produce a stable
benchmark versus a fixed opponent (default: Stockfish). It writes one JSON
summary and appends JSONL history entries for trend tracking.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import chess.pgn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nagato static Elo reference runner")
    parser.add_argument(
        "--config",
        default="training/elo_reference_config.json",
        help="Path to benchmark config JSON",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Override number of games",
    )
    parser.add_argument(
        "--tc",
        default=None,
        help="Override time control (cutechess format, e.g. 8+0.08)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override cutechess concurrency",
    )
    parser.add_argument(
        "--output",
        default="training/elo_reference_latest.json",
        help="Where to write latest result JSON",
    )
    parser.add_argument(
        "--history",
        default="training/elo_reference_history.jsonl",
        help="Where to append result history (JSONL)",
    )
    parser.add_argument(
        "--pgnout",
        default="training/elo_reference_last.pgn",
        help="Where to write match PGN",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and exit without running matches",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(path: Path, config: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def validate_paths(config: dict, root: Path) -> None:
    cutechess_path = root / config["cutechess_path"]
    if not cutechess_path.exists():
        raise FileNotFoundError(f"cutechess-cli not found: {cutechess_path}")

    for eng in config["engines"]:
        cmd_path = root / eng["cmd"]
        if not cmd_path.exists():
            raise FileNotFoundError(f"engine not found ({eng['name']}): {cmd_path}")


def engine_args(engine: dict, root: Path) -> list[str]:
    out = ["-engine", f"name={engine['name']}", f"cmd={root / engine['cmd']}", "proto=uci"]
    for arg in engine.get("args", []):
        out.append(f"arg={arg}")
    for key, value in engine.get("options", {}).items():
        out.append(f"option.{key}={value}")
    return out


def cutechess_command(config: dict, root: Path, pgnout: Path) -> list[str]:
    games = int(config["games"])
    if games <= 0:
        raise ValueError("games must be >= 1")

    cmd = [str(root / config["cutechess_path"])]
    for engine in config["engines"]:
        cmd.extend(engine_args(engine, root))

    cmd.extend(
        [
            "-each",
            f"tc={config['tc']}",
            "-games",
            str(games),
            "-repeat",
            "-concurrency",
            str(config["concurrency"]),
            "-draw",
            "movenumber=40",
            "movecount=8",
            "score=8",
            "-resign",
            "movecount=3",
            "score=700",
            "-pgnout",
            str(pgnout),
        ]
    )
    return cmd


def parse_pgn_results(pgn_path: Path, nagato_name: str) -> dict:
    w = d = l = 0
    total = 0

    with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break

            headers = game.headers
            white = headers.get("White", "")
            black = headers.get("Black", "")
            result = headers.get("Result", "*")

            if result not in {"1-0", "0-1", "1/2-1/2"}:
                continue

            if white == nagato_name:
                if result == "1-0":
                    w += 1
                elif result == "0-1":
                    l += 1
                else:
                    d += 1
                total += 1
            elif black == nagato_name:
                if result == "0-1":
                    w += 1
                elif result == "1-0":
                    l += 1
                else:
                    d += 1
                total += 1

    if total == 0:
        raise RuntimeError("No completed games found for Nagato in PGN output")

    score = w + 0.5 * d
    score_rate = score / total
    draw_rate = d / total

    if 0.0 < score_rate < 1.0:
        elo = 400.0 * math.log10(score_rate / (1.0 - score_rate))
        se = math.sqrt(score_rate * (1.0 - score_rate) / total)
        lo = max(1e-6, score_rate - 1.96 * se)
        hi = min(1.0 - 1e-6, score_rate + 1.96 * se)
        elo_lo = 400.0 * math.log10(lo / (1.0 - lo))
        elo_hi = 400.0 * math.log10(hi / (1.0 - hi))
    else:
        elo = math.inf if score_rate == 1.0 else -math.inf
        elo_lo = elo_hi = elo

    return {
        "games": total,
        "wins": w,
        "draws": d,
        "losses": l,
        "score": score,
        "score_rate": score_rate,
        "draw_rate": draw_rate,
        "elo_vs_opponent": elo,
        "elo_95ci": [elo_lo, elo_hi],
    }


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def ref_uci_elo(config: dict) -> int | None:
    if len(config.get("engines", [])) < 2:
        return None
    ref_opts = config["engines"][1].get("options", {})
    raw = ref_opts.get("UCI_Elo")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)

    config = load_config(config_path)
    run_config = copy.deepcopy(config)
    if args.games is not None:
        run_config["games"] = args.games
    if args.tc is not None:
        run_config["tc"] = args.tc
    if args.concurrency is not None:
        run_config["concurrency"] = args.concurrency

    validate_paths(run_config, root)

    pgnout = root / args.pgnout
    output_path = root / args.output
    history_path = root / args.history
    pgnout.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = cutechess_command(run_config, root, pgnout)
    print("Benchmark command:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    if pgnout.exists():
        pgnout.unlink()

    env = dict(os.environ)
    proc = subprocess.run(cmd, cwd=root, env=env, check=False)
    if proc.returncode != 0:
        print(f"cutechess-cli failed with exit code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    nagato_name = run_config["engines"][0]["name"]
    opponent_name = run_config["engines"][1]["name"]
    summary = parse_pgn_results(pgnout, nagato_name)
    summary_json = dict(summary)
    summary_json["elo_vs_opponent"] = finite_or_none(summary_json["elo_vs_opponent"])
    summary_json["elo_95ci"] = [finite_or_none(x) for x in summary_json["elo_95ci"]]

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    record = {
        "timestamp_utc": now,
        "git_commit": git_commit(root),
        "protocol": {
            "tc": run_config["tc"],
            "games_target": run_config["games"],
            "concurrency": run_config["concurrency"],
            "draw_rule": {"movenumber": 40, "movecount": 8, "score": 8},
            "resign_rule": {"movecount": 3, "score": 700},
        },
        "engines": {
            "candidate": run_config["engines"][0],
            "reference": run_config["engines"][1],
        },
        "match": summary_json,
        "files": {
            "pgn": str(pgnout.relative_to(root)),
            "latest": str(output_path.relative_to(root)),
            "history": str(history_path.relative_to(root)),
        },
    }

    # Persist run metadata back into the config for a one-file quick reference.
    ref_elo = ref_uci_elo(run_config)
    delta = summary["elo_vs_opponent"]
    est_abs = None
    if ref_elo is not None and math.isfinite(delta):
        est_abs = ref_elo + delta

    config["last_run"] = {
        "timestamp_utc": now,
        "git_commit": record["git_commit"],
        "games": summary["games"],
        "tc": run_config["tc"],
        "candidate": nagato_name,
        "reference": opponent_name,
        "reference_uci_elo": ref_elo,
        "elo_vs_reference": finite_or_none(delta),
        "elo_vs_reference_95ci": [finite_or_none(x) for x in summary["elo_95ci"]],
        "estimated_absolute_elo": est_abs,
        "wdl": {
            "wins": summary["wins"],
            "draws": summary["draws"],
            "losses": summary["losses"],
        },
    }
    save_config(config_path, config)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")

    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    elo = summary["elo_vs_opponent"]
    lo, hi = summary["elo_95ci"]
    print("\nElo reference result")
    print(f"  {nagato_name} vs {opponent_name}")
    print(
        f"  W/D/L: {summary['wins']}/{summary['draws']}/{summary['losses']} "
        f"({summary['games']} games)"
    )
    print(f"  score rate: {summary['score_rate']:.4f}, draw rate: {summary['draw_rate']:.4f}")
    print(f"  elo: {elo:.1f} (95% CI: {lo:.1f} .. {hi:.1f})")
    print(f"  latest: {output_path}")
    print(f"  history: {history_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
