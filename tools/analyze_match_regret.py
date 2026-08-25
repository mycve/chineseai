import argparse
import re
import subprocess


SCORE_RE = re.compile(r"\bscore (cp|mate) (-?\d+)")
PV_RE = re.compile(r"\bpv ([a-i][0-9][a-i][0-9])")


class Engine:
    def __init__(self, executable: str, network: str):
        self.process = subprocess.Popen(
            [executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self.send("uci")
        self.read_until("uciok")
        self.send(f"setoption name EvalFile value {network}")
        self.send("setoption name Hash value 128")
        self.send("setoption name MultiPV value 1")
        self.send("isready")
        self.read_until("readyok")

    def send(self, command: str):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_until(self, suffix: str):
        while True:
            line = self.process.stdout.readline().strip()
            if line.endswith(suffix):
                return

    def evaluate(self, position: str, depth: int, move: str | None = None):
        self.send("setoption name Clear Hash")
        self.send(position)
        command = f"go depth {depth}"
        if move:
            command += f" searchmoves {move}"
        self.send(command)
        score = None
        best = None
        while True:
            line = self.process.stdout.readline().strip()
            if line.startswith("info "):
                match = SCORE_RE.search(line)
                if match:
                    kind, value = match.groups()
                    value = int(value)
                    score = value if kind == "cp" else (30_000 if value > 0 else -30_000)
                pv = PV_RE.search(line)
                if pv:
                    best = pv.group(1)
            elif line.startswith("bestmove"):
                return score, best

    def close(self):
        self.send("quit")
        self.process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("network")
    parser.add_argument("fen")
    parser.add_argument("moves")
    parser.add_argument("--chinese", choices=("red", "black"), required=True)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--threshold", type=int, default=30)
    args = parser.parse_args()
    moves = args.moves.split()
    initial_red = args.fen.split()[1] == "w"
    engine = Engine(args.engine, args.network)
    rows = []
    try:
        for ply, played in enumerate(moves):
            red_to_move = initial_red == (ply % 2 == 0)
            chinese_to_move = red_to_move == (args.chinese == "red")
            if not chinese_to_move:
                continue
            prefix = " ".join(moves[:ply])
            position = f"position fen {args.fen}" + (f" moves {prefix}" if prefix else "")
            best_score, best_move = engine.evaluate(position, args.depth)
            played_score, _ = engine.evaluate(position, args.depth, played)
            regret = best_score - played_score
            rows.append((ply + 1, played, best_move, best_score, played_score, regret))
    finally:
        engine.close()
    print("ply,played,best,best_cp,played_cp,regret_cp")
    for row in rows:
        if row[-1] >= args.threshold or row[1] != row[2]:
            print(",".join(map(str, row)))
    serious = [row for row in rows if row[-1] >= args.threshold]
    print(f"summary decisions={len(rows)} serious={len(serious)} threshold={args.threshold}cp")


if __name__ == "__main__":
    main()
