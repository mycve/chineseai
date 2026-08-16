import argparse
import json
import struct
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def load_data(path):
    raw = memoryview(Path(path).read_bytes())
    if raw[:4] != b"XQPF":
        raise ValueError("bad dataset magic")
    version, count = struct.unpack_from("<II", raw, 4)
    if version != 1:
        raise ValueError(f"unsupported dataset version {version}")
    offset, rows, max_features, max_moves = 12, [], 0, 0
    for _ in range(count):
        nf, nm, best = struct.unpack_from("<HHH", raw, offset)
        offset += 6
        wdl = struct.unpack_from("<fff", raw, offset)
        offset += 12
        features = struct.unpack_from(f"<{nf}H", raw, offset)
        offset += nf * 2
        moves = struct.unpack_from(f"<{nm}H", raw, offset)
        offset += nm * 2
        rows.append((features, moves, best, wdl))
        max_features, max_moves = max(max_features, nf), max(max_moves, nm)
    feature_vocab = max(max(r[0]) for r in rows) + 1
    move_vocab = max(max(r[1]) for r in rows) + 1
    features = torch.full((count, max_features), feature_vocab, dtype=torch.int64)
    moves = torch.zeros((count, max_moves), dtype=torch.int64)
    mask = torch.zeros((count, max_moves), dtype=torch.bool)
    best = torch.empty(count, dtype=torch.int64)
    wdl = torch.empty((count, 3), dtype=torch.float32)
    for i, (fs, ms, target, value) in enumerate(rows):
        features[i, :len(fs)] = torch.tensor(fs)
        moves[i, :len(ms)] = torch.tensor(ms)
        mask[i, :len(ms)] = True
        best[i], wdl[i] = target, torch.tensor(value)
    return features, moves, mask, best, wdl, feature_vocab, move_vocab


class PolicyNet(nn.Module):
    def __init__(self, variant, feature_vocab, move_vocab, hidden=128, rank=32):
        super().__init__()
        self.variant, self.rank = variant, rank
        self.feature = nn.Embedding(feature_vocab + 1, hidden, padding_idx=feature_vocab)
        self.norm = nn.RMSNorm(hidden)
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.move_bias = nn.Embedding(move_vocab, 1)
        self.move = nn.Embedding(move_vocab, rank)
        self.query = nn.Linear(hidden, rank)
        if variant == "gated":
            self.gate = nn.Linear(hidden, rank)
        elif variant == "hyper2":
            self.move2 = nn.Embedding(move_vocab, rank)
            self.query2 = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, rank))
        elif variant in ("interaction_mlp", "residual_mlp"):
            self.policy_mlp = nn.Sequential(nn.Linear(rank * 3, 64), nn.SiLU(), nn.Linear(64, 1))
        elif variant == "product_mlp":
            self.policy_mlp = nn.Sequential(nn.Linear(rank, 64), nn.SiLU(), nn.Linear(64, 1))
        elif variant != "dot":
            raise ValueError(variant)
        self.value = nn.Sequential(nn.Linear(hidden, 96), nn.SiLU(), nn.Linear(96, 3))

    def forward(self, features, moves, mask):
        context = self.trunk(self.norm(self.feature(features).sum(1)))
        move = self.move(moves)
        query = self.query(context)
        if self.variant == "dot":
            logits = (move * query[:, None]).sum(-1)
        elif self.variant == "gated":
            logits = (move * query[:, None] * (2 * torch.sigmoid(self.gate(context)))[:, None]).sum(-1)
        elif self.variant == "hyper2":
            logits = (move * query[:, None]).sum(-1)
            logits += (self.move2(moves) * self.query2(context)[:, None]).sum(-1)
        elif self.variant in ("interaction_mlp", "residual_mlp"):
            q = query[:, None].expand_as(move)
            logits = self.policy_mlp(torch.cat((q, move, q * move), -1)).squeeze(-1)
            if self.variant == "residual_mlp":
                logits += (q * move).sum(-1)
        else:
            logits = self.policy_mlp(query[:, None] * move).squeeze(-1)
        logits = logits + self.move_bias(moves).squeeze(-1)
        return logits.masked_fill(~mask, -1e4), self.value(context)


@torch.inference_mode()
def evaluate(model, tensors, indices, batch_size):
    model.eval()
    total_ce = total_top1 = total_top3 = total_value = 0.0
    features, moves, mask, best, wdl = tensors
    for ids in indices.split(batch_size):
        logits, value = model(features[ids], moves[ids], mask[ids])
        total_ce += F.cross_entropy(logits.float(), best[ids], reduction="sum").item()
        order = logits.topk(3, dim=1).indices
        total_top1 += (order[:, 0] == best[ids]).sum().item()
        total_top3 += (order == best[ids, None]).any(1).sum().item()
        total_value += (-(wdl[ids] * F.log_softmax(value.float(), -1)).sum(-1)).sum().item()
    n = len(indices)
    return {"policy_ce": total_ce / n, "top1": total_top1 / n,
            "top3": total_top3 / n, "value_ce": total_value / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--variants", default="dot,gated,hyper2,interaction_mlp")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", default="eval/torch-policy-search")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    loaded = load_data(args.dataset)
    *cpu_tensors, feature_vocab, move_vocab = loaded
    tensors = tuple(t.to(device) for t in cpu_tensors)
    count = len(cpu_tensors[0])
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(args.seed)).to(device)
    split = count - max(1, count // 10)
    train_ids, validation_ids = permutation[:split], permutation[split:]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in args.variants.split(","):
        torch.manual_seed(args.seed)
        model = PolicyNet(variant, feature_vocab, move_vocab).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        model.train()
        steps = processed = 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        for ids in train_ids.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, value = model(*(t[ids] for t in tensors[:3]))
                loss = F.cross_entropy(logits.float(), tensors[3][ids])
                loss += .25 * (-(tensors[4][ids] * F.log_softmax(value.float(), -1)).sum(-1)).mean()
            loss.backward(); optimizer.step()
            steps += 1; processed += len(ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = evaluate(model, tensors, validation_ids, args.batch_size)
        row = {"variant": variant, "parameters": sum(p.numel() for p in model.parameters()),
               "seconds": elapsed, "steps": steps, "processed": processed,
               "epochs": 1, "samples_per_second": processed / elapsed, **metrics}
        results.append(row)
        torch.save(model.state_dict(), output / f"{variant}.pt")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
