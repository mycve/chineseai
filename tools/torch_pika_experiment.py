import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_split(prefix: str):
    meta = json.loads(Path(f"{prefix}.meta.json").read_text(encoding="utf-8"))
    n = meta["samples"]
    features = np.memmap(
        meta["features"], dtype=np.int32, mode="r", shape=(n, meta["max_features"])
    )
    legal = np.memmap(
        meta["legal_moves"], dtype=np.int32, mode="r", shape=(n, meta["max_legal_moves"])
    )
    targets = np.memmap(meta["target_moves"], dtype=np.int64, mode="r", shape=(n,))
    values = np.memmap(meta["values"], dtype=np.float32, mode="r", shape=(n,))
    return meta, features, legal, targets, values


class PikaDataset(torch.utils.data.Dataset):
    def __init__(self, prefix: str, preload: bool = True):
        self.meta, self.features, self.legal, self.targets, self.values = load_split(prefix)
        if preload:
            self.features = torch.from_numpy((np.asarray(self.features, dtype=np.int64) + 1).copy())
            self.legal = torch.from_numpy(np.asarray(self.legal, dtype=np.int64).copy())
            self.targets = torch.from_numpy(np.asarray(self.targets, dtype=np.int64).copy())
            self.values = torch.from_numpy(np.asarray(self.values, dtype=np.float32).copy())

    def __len__(self):
        return self.meta["samples"]

    def __getitem__(self, idx):
        if isinstance(self.features, torch.Tensor):
            return self.features[idx], self.legal[idx], self.targets[idx], self.values[idx]
        return (
            torch.from_numpy(np.asarray(self.features[idx], dtype=np.int64) + 1),
            torch.from_numpy(np.asarray(self.legal[idx], dtype=np.int64)),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.values[idx], dtype=torch.float32),
        )


class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, mult: int = 2, dropout: float = 0.0):
        super().__init__()
        inner = hidden * mult
        self.fc1 = nn.Linear(hidden, inner)
        self.fc2 = nn.Linear(inner, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm()

    def forward(self, x):
        y = self.fc2(self.dropout(F.silu(self.fc1(x))))
        return self.norm(x + y)


class PikaNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        move_space: int,
        hidden: int,
        blocks: int,
        block_mult: int,
        dropout: float,
        value_head: int,
        tower: str,
        policy_blocks: int,
        value_blocks: int,
    ):
        super().__init__()
        self.move_space = move_space
        self.tower = tower
        self.embed = nn.Embedding(input_size + 1, hidden, padding_idx=0)
        self.input_bias = nn.Parameter(torch.zeros(hidden))
        self.input_norm = RMSNorm()
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, block_mult, dropout) for _ in range(blocks)]
        )
        self.policy_tower = nn.Sequential(
            *[ResidualBlock(hidden, block_mult, dropout) for _ in range(policy_blocks)]
        )
        self.value_tower = nn.Sequential(
            *[ResidualBlock(hidden, block_mult, dropout) for _ in range(value_blocks)]
        )
        self.policy = nn.Linear(hidden, move_space)
        self.value = nn.Sequential(
            nn.Linear(hidden, value_head),
            nn.SiLU(),
            nn.Linear(value_head, value_head),
            nn.SiLU(),
            nn.Linear(value_head, 1),
            nn.Tanh(),
        )
        nn.init.normal_(self.embed.weight, std=0.015)
        with torch.no_grad():
            self.embed.weight[0].zero_()

    def forward(self, features, legal):
        mask = features.ne(0).unsqueeze(-1)
        hidden = (self.embed(features) * mask).sum(dim=1) + self.input_bias
        hidden = self.input_norm(F.relu(hidden))
        hidden = self.blocks(hidden)
        policy_hidden = self.policy_tower(hidden) if self.tower == "dual" else hidden
        value_hidden = self.value_tower(hidden) if self.tower == "dual" else hidden
        logits = self.policy(policy_hidden)
        legal_mask_with_pad = torch.zeros(
            (features.shape[0], self.move_space + 1), dtype=torch.bool, device=features.device
        )
        safe_legal = (legal + 1).clamp_min(0)
        legal_mask_with_pad.scatter_(1, safe_legal, True)
        legal_mask = legal_mask_with_pad[:, 1:]
        logits = logits.masked_fill(~legal_mask, -1.0e9)
        value = self.value(value_hidden).squeeze(-1)
        return logits, value


def make_loader(dataset, batch_size, shuffle, workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    policy_sum = 0.0
    value_sum = 0.0
    top1 = 0
    top4 = 0
    for features, legal, targets, values in loader:
        features = features.to(device, non_blocking=True)
        legal = legal.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        logits, pred_values = model(features, legal)
        policy_loss = F.cross_entropy(logits, targets, reduction="sum")
        value_loss = F.mse_loss(pred_values, values, reduction="sum")
        k = min(4, logits.shape[1])
        top = logits.topk(k, dim=1).indices
        top1 += top[:, 0].eq(targets).sum().item()
        top4 += top.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
        n = features.shape[0]
        total += n
        policy_sum += policy_loss.item()
        value_sum += value_loss.item()
    return {
        "policy_ce": policy_sum / total,
        "value_mse": value_sum / total,
        "top1": top1 / total,
        "top4": top4 / total,
    }


def train_one(config, train_ds, valid_ds, args):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = PikaNet(
        input_size=train_ds.meta["input_size"],
        move_space=train_ds.meta["move_space"],
        hidden=config["hidden"],
        blocks=config["blocks"],
        block_mult=config["mult"],
        dropout=args.dropout,
        value_head=args.value_head,
        tower=config["tower"],
        policy_blocks=config["policy_blocks"],
        value_blocks=config["value_blocks"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, args.batch_size, True, args.workers)
    valid_loader = make_loader(valid_ds, args.eval_batch_size, False, args.workers)
    initial = evaluate(model, valid_loader, device)
    started = time.time()
    model.train()
    seen = 0
    train_policy = 0.0
    train_value = 0.0
    for features, legal, targets, values in train_loader:
        features = features.to(device, non_blocking=True)
        legal = legal.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        logits, pred_values = model(features, legal)
        policy_loss = F.cross_entropy(logits, targets)
        value_loss = F.mse_loss(pred_values, values)
        loss = policy_loss + args.value_weight * value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        n = features.shape[0]
        seen += n
        train_policy += policy_loss.item() * n
        train_value += value_loss.item() * n
    elapsed = time.time() - started
    final = evaluate(model, valid_loader, device)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"{config['name']}: params={params/1e6:.2f}M device={device} "
        f"init_ce={initial['policy_ce']:.4f} final_ce={final['policy_ce']:.4f} "
        f"top1={final['top1']:.4f} top4={final['top4']:.4f} "
        f"value_mse={final['value_mse']:.4f} "
        f"train_ce={train_policy/seen:.4f} train_value={train_value/seen:.4f} "
        f"elapsed={elapsed:.1f}s"
    )
    return final


def parse_experiment(text):
    # name:hidden:blocks:mult:seed[:tower[:policy_blocks[:value_blocks]]]
    parts = text.split(":")
    if len(parts) not in (5, 6, 7, 8):
        raise ValueError(
            f"bad experiment '{text}', expected name:hidden:blocks:mult:seed[:tower[:policy_blocks[:value_blocks]]]"
        )
    tower = parts[5] if len(parts) >= 6 else "shared"
    return {
        "name": parts[0],
        "hidden": int(parts[1]),
        "blocks": int(parts[2]),
        "mult": int(parts[3]),
        "seed": int(parts[4]),
        "tower": tower,
        "policy_blocks": int(parts[6]) if len(parts) >= 7 else 0,
        "value_blocks": int(parts[7]) if len(parts) >= 8 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="eval/torch/pika_train200k")
    parser.add_argument("--valid", default="eval/torch/pika_valid50k")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--value-head", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "h192_b0:192:0:2:20260710",
            "h192_b1:192:1:2:20260711",
            "h192_b2:192:2:2:20260712",
            "h320_b0:320:0:2:20260713",
            "h320_b1:320:1:2:20260714",
        ],
    )
    args = parser.parse_args()
    train_ds = PikaDataset(args.train, preload=not args.no_preload)
    valid_ds = PikaDataset(args.valid, preload=not args.no_preload)
    print(
        f"data: train={len(train_ds)} valid={len(valid_ds)} "
        f"input={train_ds.meta['input_size']} moves={train_ds.meta['move_space']} "
        f"cuda={torch.cuda.is_available()}"
    )
    for item in args.experiments:
        train_one(parse_experiment(item), train_ds, valid_ds, args)


if __name__ == "__main__":
    main()
