"""从本地成对子力表生成新残局候选，合法性由Rust查询器复核。Python标准库，无网络。"""
import argparse
import random
from pathlib import Path


def squares(piece, red):
    if piece == "K":
        points = [(r, f) for r in [0, 1, 2] for f in [3, 4, 5]]
    elif piece == "A":
        points = [(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)]
    elif piece == "B":
        points = [(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6)]
    elif piece == "P":
        points = [(r, f) for r in range(5, 10) for f in range(9)] + [(r, f) for r in [3, 4] for f in [0, 2, 4, 6, 8]]
    else:
        points = [(r, f) for r in range(10) for f in range(9)]
    return [(9-r if red else r, f) for r, f in points]


def candidate(material, rng):
    split = material.index("K", 1)
    board = {}
    for red, pieces in [(True, material[:split]), (False, material[split:])]:
        for piece in pieces:
            free = [s for s in squares(piece, red) if s not in board]
            if not free:
                return None
            board[rng.choice(free)] = piece if red else piece.lower()
    red_king = next(s for s, p in board.items() if p == "K")
    black_king = next(s for s, p in board.items() if p == "k")
    if red_king[1] == black_king[1] and not any(
        (r, red_king[1]) in board for r in range(black_king[0] + 1, red_king[0])
    ):
        return None
    rows = []
    for r in range(10):
        line, empty = "", 0
        for f in range(9):
            if (r, f) not in board:
                empty += 1
            else:
                if empty:
                    line += str(empty)
                    empty = 0
                line += board[r, f]
        if empty:
            line += str(empty)
        rows.append(line)
    return "/".join(rows) + " " + rng.choice(["w", "b"]) + " - - 0 1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path("data/tablebases/chessdb"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-material", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    materials = sorted({p.stem for p in (args.tables / "EGTB_DTM").glob("*.lzdtm")} & {p.stem for p in (args.tables / "EGTB_DTC").glob("*.lzdtc")})
    if not materials or args.per_material < 1:
        raise ValueError("没有成对表，或采样数无效")
    rng, rows = random.Random(args.seed), []
    for material in materials:
        rows.append("# " + material)
        accepted = 0
        for _ in range(args.per_material * 100):
            fen = candidate(material, rng)
            if fen:
                rows.append(fen)
                accepted += 1
                if accepted == args.per_material:
                    break
        if accepted != args.per_material:
            raise RuntimeError("生成候选不足：" + material)
    with args.output.open("x", encoding="utf-8") as file:
        file.write("\n".join(rows) + "\n")
    print(f"materials={len(materials)}, candidates={sum(not s.startswith('#') for s in rows)}")


if __name__ == "__main__":
    main()
