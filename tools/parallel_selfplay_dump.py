#!/usr/bin/env python3

import argparse
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool


def run_chunk(job: tuple[int, int, int, int, str, bool, str, str | None]) -> tuple[int, pathlib.Path, int]:
    index, games, depth, max_plies, root_text, release, temp_dir_text, engine_text = job
    root = pathlib.Path(root_text)
    temp_dir = pathlib.Path(temp_dir_text)
    output = temp_dir / f"selfplay_chunk_{index:04}.txt"
    if engine_text is not None:
        command = [engine_text, "selfplay-dump", str(games), str(depth), str(output), str(max_plies)]
    else:
        command = ["cargo", "run"]
        if release:
            command.append("--release")
        command.extend(["--", "selfplay-dump", str(games), str(depth), str(output), str(max_plies)])
    subprocess.run(command, cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    line_count = sum(1 for _ in output.open("r", encoding="utf-8"))
    return index, output, line_count


def resolve_worker_count(workers: str, reserve: int, max_workers: int | None) -> int:
    if workers == "auto":
        count = max(1, (os.cpu_count() or 1) - max(0, reserve))
        if max_workers is not None:
            count = min(count, max_workers)
        return count
    try:
        return max(1, int(workers))
    except ValueError as exc:
        raise SystemExit("--workers must be an integer or 'auto'") from exc


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate ChineseAI self-play FEN rows in parallel."
    )
    parser.add_argument("output", type=pathlib.Path, nargs="?", default=root / "data" / "selfplay.fens.txt")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--auto-worker-reserve", type=int, default=2)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--release", action="store_true")
    parser.add_argument(
        "--engine",
        type=pathlib.Path,
        help="prebuilt ChineseAI executable; defaults to target/release/chineseai when present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    workers = min(
        resolve_worker_count(args.workers, args.auto_worker_reserve, args.max_workers),
        max(1, args.games),
    )
    engine = resolve_engine(root, args.engine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    games_per_worker = math.ceil(args.games / workers)

    jobs = []
    remaining = args.games
    with tempfile.TemporaryDirectory(prefix="chineseai_selfplay_") as temp_dir:
        for index in range(workers):
            games = min(games_per_worker, remaining)
            if games <= 0:
                break
            remaining -= games
            jobs.append((index, games, args.depth, args.max_plies, str(root), args.release, temp_dir, engine))

        print(
            f"parallel selfplay: games={args.games} workers={len(jobs)} "
            f"depth={args.depth} max_plies={args.max_plies} "
            f"runner={engine or 'cargo run'}",
            flush=True,
        )
        if len(jobs) == 1:
            results = [run_chunk(jobs[0])]
        else:
            with Pool(processes=len(jobs)) as pool:
                results = list(pool.imap_unordered(run_chunk, jobs))

        rows = 0
        with args.output.open("w", encoding="utf-8") as out:
            for index, chunk_path, line_count in sorted(results):
                out.write(chunk_path.read_text(encoding="utf-8"))
                rows += line_count
                print(
                    f"merged chunk={index} rows={line_count} total_rows={rows} -> {args.output}",
                    flush=True,
                )
    return 0


def resolve_engine(root: pathlib.Path, engine_arg: pathlib.Path | None) -> str | None:
    if engine_arg is not None:
        engine = engine_arg.resolve()
        if not engine.exists():
            raise SystemExit(f"engine not found: {engine}")
        return str(engine)

    release_engine = root / "target" / "release" / "chineseai"
    if release_engine.exists():
        return str(release_engine)

    if shutil.which("cargo") is None:
        raise SystemExit(
            "cargo not found and target/release/chineseai does not exist. "
            "Build or copy the engine first, or pass --engine /path/to/chineseai."
        )
    return None


if __name__ == "__main__":
    sys.exit(main())
