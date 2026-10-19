"""显式将 v39 检查点迁移到 v40；删除未被激活的车炮配对行并复制 value 输出桶。"""
import argparse
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file, save_file


def reachable(piece, square):
    rank, file = divmod(square, 9)
    ours, kind = piece < 7, piece % 7
    if kind == 0:
        return 3 <= file <= 5 and (rank >= 7 if ours else rank <= 2)
    if kind == 1:
        return (rank, file) in ([(7, 3), (7, 5), (8, 4), (9, 3), (9, 5)] if ours else [(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)])
    if kind == 2:
        return (rank, file) in ([(5, 2), (5, 6), (7, 0), (7, 4), (7, 8), (9, 2), (9, 6)] if ours else [(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6)])
    if kind == 6:
        return (rank <= 4 or rank in (5, 6) and file % 2 == 0) if ours else (rank >= 5 or rank in (3, 4) and file % 2 == 0)
    return True


def retained_rows():
    rows, offset = [], 0
    for attacker in range(14):
        ours, kind = attacker < 7, attacker % 7
        for source in range(90):
            if not reachable(attacker, source):
                continue
            rank, file = divmod(source, 9)
            if kind in (4, 5):
                targets = [t for t in range(90) if t != source and (t // 9 == rank or t % 9 == file)]
            else:
                steps = {0: [(0, -1), (0, 1), (-1, 0), (1, 0)], 1: [(-1, -1), (1, -1), (-1, 1), (1, 1)], 2: [(-2, -2), (2, -2), (-2, 2), (2, 2)], 3: [(-1, -2), (1, -2), (-1, 2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1)], 6: ([(0, -1), (-1, 0), (1, 0)] if ours and rank <= 4 else [(0, 1), (-1, 0), (1, 0)] if not ours and rank >= 5 else [(0, -1)] if ours else [(0, 1)])}[kind]
                targets = []
                for df, dr in steps:
                    f, r = file + df, rank + dr
                    if not (0 <= f < 9 and 0 <= r < 10):
                        continue
                    if kind in (0, 1) and (not 3 <= f <= 5 or (r < 7 if ours else r > 2)):
                        continue
                    if kind == 2 and (r < 5 if ours else r > 4):
                        continue
                    targets.append(r * 9 + f)
            for target in targets:
                count = sum(reachable(p, target) for p in range(14))
                if kind not in (4, 5):
                    rows.extend(range(offset, offset + count))
                offset += count
    assert offset == 57702
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    tensors = load_file(args.input)
    assert tensors["az_model_format_version"].item() == 39
    rows = retained_rows()
    table = tensors["value_threat_embedding"]
    tensors["value_threat_embedding"] = np.concatenate((table[rows], table[57702:]), axis=0)
    tensors["value_head_output"] = np.tile(tensors["value_head_output"], (32, 1)).reshape(32, 3, 96)
    tensors["az_model_format_version"] = np.array([40.0], dtype=np.float32)
    save_file(tensors, args.output)
    print(f"pair rows: 57702 -> {len(rows)}; value buckets: 32")
