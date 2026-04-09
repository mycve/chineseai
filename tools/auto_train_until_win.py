#!/usr/bin/env python3

import argparse
import json
import pathlib
import subprocess
import sys
import time


def run(command: list[str], cwd: pathlib.Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def require_path(path: pathlib.Path, label: str) -> pathlib.Path:
    path = path.resolve()
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    return path


def score_from_report(report_path: pathlib.Path, label: str) -> tuple[float, int]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    games = int(report["games"])
    score = float(report["score"])
    results = report["results"]
    print(
        f"match result: score={score:.1f}/{games} rate={score / max(games, 1):.3f} "
        f"{label}={results.get(label, 0)} pikafish={results.get('pikafish', 0)} draw={results.get('draw', 0)}",
        flush=True,
    )
    return score / max(games, 1), games


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Loop: self-play positions -> Pikafish distill -> train NNUE -> match; stop at target score."
    )
    parser.add_argument("--work-dir", type=pathlib.Path, default=root / "runs" / "auto_train")
    parser.add_argument("--engine", type=pathlib.Path, default=root / "target" / "release" / "chineseai")
    parser.add_argument("--pikafish", type=pathlib.Path, default=root / "tools" / "pikafish")
    parser.add_argument("--pikafish-nnue", type=pathlib.Path, default=root / "tools" / "pikafish.nnue")
    parser.add_argument("--start-model", type=pathlib.Path, help="resume from an existing ChineseAI text NNUE model")
    parser.add_argument("--max-iters", type=int, default=1000000)
    parser.add_argument("--target-score", type=float, default=0.50, help="stop when match score rate reaches this value")

    parser.add_argument("--games", type=int, default=2000, help="self-play games per iteration")
    parser.add_argument("--selfplay-depth", type=int, default=1)
    parser.add_argument("--selfplay-nodes", type=int, default=4_000)
    parser.add_argument("--selfplay-random-plies", type=int, default=6)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--selfplay-workers", default="auto")

    parser.add_argument("--teacher-movetime-ms", type=int, default=2)
    parser.add_argument("--teacher-depth", type=int)
    parser.add_argument("--distill-workers", default="auto")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--auto-worker-reserve", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=96)
    parser.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--feature-set", choices=["v2"], default="v2")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--validation-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260409)

    parser.add_argument("--match-games", type=int, default=40)
    parser.add_argument("--match-movetime-ms", type=int, default=2)
    parser.add_argument("--match-max-plies", type=int, default=120)
    parser.add_argument("--match-workers", default="auto")
    parser.add_argument("--match-max-workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--match-quiet", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    engine = require_path(args.engine, "ChineseAI engine")
    pikafish = require_path(args.pikafish, "Pikafish engine")
    pikafish_nnue = require_path(args.pikafish_nnue, "Pikafish NNUE")
    previous_model = args.start_model.resolve() if args.start_model else None
    if previous_model is not None:
        previous_model = require_path(previous_model, "start model")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "data").mkdir(exist_ok=True)
    (args.work_dir / "models").mkdir(exist_ok=True)
    (args.work_dir / "reports").mkdir(exist_ok=True)

    print(
        f"auto train start: work_dir={args.work_dir} target_score={args.target_score:.3f} "
        f"engine={engine} pikafish={pikafish}",
        flush=True,
    )

    best_score = -1.0
    best_model = previous_model
    started = time.monotonic()

    for iteration in range(1, args.max_iters + 1):
        prefix = f"iter_{iteration:04}"
        fen_path = args.work_dir / "data" / f"{prefix}.fens.txt"
        sample_path = args.work_dir / "data" / f"{prefix}.samples.txt"
        model_path = args.work_dir / "models" / f"{prefix}.nnue.txt"
        report_path = args.work_dir / "reports" / f"{prefix}.match.json"

        print(f"\n=== iteration {iteration} ===", flush=True)

        selfplay_cmd = [
            "python3",
            "tools/parallel_selfplay_dump.py",
            str(fen_path),
            "--games",
            str(args.games),
            "--depth",
            str(args.selfplay_depth),
            "--max-plies",
            str(args.max_plies),
            "--nodes",
            str(args.selfplay_nodes),
            "--random-plies",
            str(args.selfplay_random_plies),
            "--workers",
            str(args.selfplay_workers),
            "--auto-worker-reserve",
            str(args.auto_worker_reserve),
            "--engine",
            str(engine),
        ]
        if args.max_workers is not None:
            selfplay_cmd.extend(["--max-workers", str(args.max_workers)])
        run(selfplay_cmd, root)

        distill_cmd = [
            "python3",
            "tools/parallel_distill_pikafish.py",
            str(fen_path),
            str(sample_path),
            "--feature-set",
            args.feature_set,
            "--workers",
            str(args.distill_workers),
            "--chunk-size",
            str(args.chunk_size),
            "--auto-worker-reserve",
            str(args.auto_worker_reserve),
            "--pikafish",
            str(pikafish),
            "--pikafish-nnue",
            str(pikafish_nnue),
        ]
        if args.max_workers is not None:
            distill_cmd.extend(["--max-workers", str(args.max_workers)])
        if args.teacher_depth is not None:
            distill_cmd.extend(["--depth", str(args.teacher_depth)])
        else:
            distill_cmd.extend(["--movetime-ms", str(args.teacher_movetime_ms)])
        if args.dedup:
            distill_cmd.append("--dedup")
        run(distill_cmd, root)

        train_cmd = [
            "python3",
            "tools/train_nnue.py",
            str(sample_path),
            str(model_path),
            "--feature-set",
            args.feature_set,
            "--hidden-size",
            str(args.hidden_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--validation-split",
            str(args.validation_split),
            "--seed",
            str(args.seed + iteration),
        ]
        if previous_model is not None:
            train_cmd.extend(["--resume", str(previous_model)])
        run(train_cmd, root)

        match_cmd = [
            "python3",
            "tools/run_matches.py",
            "--games",
            str(args.match_games),
            "--movetime-ms",
            str(args.match_movetime_ms),
            "--max-plies",
            str(args.match_max_plies),
            "--workers",
            str(args.match_workers),
            "--auto-worker-reserve",
            str(args.auto_worker_reserve),
            "--progress-every",
            str(args.progress_every),
            "--ours",
            str(engine),
            "--pikafish",
            str(pikafish),
            "--pikafish-nnue",
            str(pikafish_nnue),
            "--ours-eval",
            str(model_path),
            "--report-out",
            str(report_path),
            "--label",
            "candidate",
        ]
        if args.match_max_workers is not None:
            match_cmd.extend(["--max-workers", str(args.match_max_workers)])
        if args.match_quiet:
            match_cmd.append("--quiet")
        run(match_cmd, root)

        score_rate, games = score_from_report(report_path, "candidate")
        previous_model = model_path
        if score_rate > best_score:
            best_score = score_rate
            best_model = model_path
            (args.work_dir / "models" / "best_model_path.txt").write_text(str(best_model) + "\n")
            print(f"new best: {best_score:.3f} model={best_model}", flush=True)

        if score_rate >= args.target_score:
            print(
                f"target reached: score_rate={score_rate:.3f} games={games} "
                f"model={model_path} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
            return 0

        print(
            f"continue: score_rate={score_rate:.3f} < target={args.target_score:.3f} "
            f"best={best_score:.3f}",
            flush=True,
        )

    print(f"max iterations reached. best={best_score:.3f} model={best_model}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
