import argparse
import json
import struct
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open


ADVISOR = {(0, 3), (0, 5), (1, 4), (2, 3), (2, 5), (7, 3), (7, 5), (8, 4), (9, 3), (9, 5)}
ELEPHANT = {(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6),
            (5, 2), (5, 6), (7, 0), (7, 4), (7, 8), (9, 2), (9, 6)}


def dense_move_squares():
    pairs = []
    for source in range(90):
        sr, sf = divmod(source, 9)
        for target in range(90):
            if source == target:
                continue
            tr, tf = divmod(target, 9)
            dr, df = abs(tr - sr), abs(tf - sf)
            valid = df == 0 or dr == 0 or (df, dr) in ((1, 2), (2, 1))
            valid |= df == dr == 1 and (sr, sf) in ADVISOR and (tr, tf) in ADVISOR
            valid |= df == dr == 2 and (sr, sf) in ELEPHANT and (tr, tf) in ELEPHANT
            if valid:
                pairs.append((source, target))
    return pairs


def load_data(path):
    raw = memoryview(Path(path).read_bytes())
    if raw[:4] != b"XQPF":
        raise ValueError("bad dataset magic")
    version, count = struct.unpack_from("<II", raw, 4)
    if version not in (1, 2, 3):
        raise ValueError(f"unsupported dataset version {version}")
    offset, rows, max_features, max_moves = 12, [], 0, 0
    for row_index in range(count):
        if version >= 3:
            group, ply = struct.unpack_from("<IB", raw, offset)
            offset += 5
        else:
            group, ply = row_index, 0
        nf, nm, best = struct.unpack_from("<HHH", raw, offset)
        offset += 6
        wdl = struct.unpack_from("<fff", raw, offset)
        offset += 12
        features = struct.unpack_from(f"<{nf}H", raw, offset)
        offset += nf * 2
        moves = struct.unpack_from(f"<{nm}H", raw, offset)
        offset += nm * 2
        checks = tuple(raw[offset:offset + nm]) if version >= 2 else (0,) * nm
        offset += nm if version >= 2 else 0
        rows.append((features, moves, checks, best, wdl, group, ply))
        max_features, max_moves = max(max_features, nf), max(max_moves, nm)
    feature_vocab = max(max(r[0]) for r in rows) + 1
    move_vocab = max(max(r[1]) for r in rows) + 1
    features = torch.full((count, max_features), feature_vocab, dtype=torch.int64)
    board = torch.zeros((count, 90), dtype=torch.int64)
    moves = torch.zeros((count, max_moves), dtype=torch.int64)
    checks = torch.zeros((count, max_moves), dtype=torch.int64)
    mask = torch.zeros((count, max_moves), dtype=torch.bool)
    best = torch.empty(count, dtype=torch.int64)
    wdl = torch.empty((count, 3), dtype=torch.float32)
    groups = torch.empty(count, dtype=torch.int64)
    plies = torch.empty(count, dtype=torch.int64)
    for i, (fs, ms, cs, target, value, group, ply) in enumerate(rows):
        features[i, :len(fs)] = torch.tensor(fs)
        for feature in fs:
            board[i, feature % 90] = feature // 90 + 1
        moves[i, :len(ms)] = torch.tensor(ms)
        checks[i, :len(cs)] = torch.tensor(cs)
        mask[i, :len(ms)] = True
        best[i], wdl[i] = target, torch.tensor(value)
        groups[i], plies[i] = group, ply
    return features, board, moves, checks, mask, best, wdl, groups, plies, feature_vocab, move_vocab


class ResidualTrunk(nn.Module):
    def __init__(self, hidden, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(),
                                                  nn.Linear(hidden * 2, hidden)) for _ in range(depth))
        self.scales = nn.Parameter(torch.zeros(depth))

    def forward(self, value):
        for scale, block in zip(self.scales, self.blocks):
            value = value + scale * block(F.rms_norm(value, (value.shape[-1],)))
        return value


class PolicyNet(nn.Module):
    def __init__(self, variant, feature_vocab, move_vocab, hidden=128, rank=32):
        super().__init__()
        self.variant, self.rank = variant, rank
        self.feature = nn.Embedding(feature_vocab + 1, hidden, padding_idx=feature_vocab)
        self.norm = nn.RMSNorm(hidden)
        self.trunk = (ResidualTrunk(hidden, 8 if "verydeep" in variant else 4)
                      if "deep" in variant else nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()))
        if "conv" in variant:
            channels = 64
            self.board_piece = nn.Embedding(15, channels)
            blocks = []
            for dilation in (1, 2, 4, 1, 2, 4):
                blocks.extend((nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation), nn.GELU()))
            self.board_conv = nn.Sequential(*blocks)
            self.board_proj = nn.Linear(channels * 90, hidden)
            self.move_local = nn.Linear(channels * 2, rank)
        if "cross" in variant:
            self.cross_piece = nn.Linear(hidden, rank)
            self.cross_attention = nn.MultiheadAttention(rank, 4, batch_first=True)
        if "attention" in variant:
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, 0.0, "gelu", batch_first=True, norm_first=True)
            self.board_attention = nn.TransformerEncoder(layer, 2 if "attention2" in variant else 1)
        self.move_bias = nn.Embedding(move_vocab, 1)
        self.move = nn.Embedding(move_vocab, rank)
        self.query = nn.Linear(hidden, rank)
        if variant.startswith("full_accumulator"):
            accumulator_input = hidden + rank
            self.accumulator_move = nn.Embedding(move_vocab, rank)
            self.accumulator_check = nn.Embedding(2, rank)
            if "piece_moe" in variant:
                self.piece_moe_experts = nn.ModuleList(
                    nn.Sequential(nn.Linear(policy_input, rank * 2), nn.SiLU(), nn.Linear(rank * 2, 1))
                    for _ in range(8))
            elif "moe" in variant:
                self.moe_gate = nn.Linear(accumulator_input, 8)
                self.moe_experts = nn.ModuleList(
                    nn.Sequential(nn.Linear(accumulator_input, hidden * 2), nn.SiLU(),
                                  nn.Linear(hidden * 2, 1)) for _ in range(8))
            else:
                depth = 3 if "deep" in variant else 1
                layers = [nn.Linear(accumulator_input, hidden * 2), nn.SiLU()]
                for _ in range(depth - 1):
                    layers.extend((nn.Linear(hidden * 2, hidden * 2), nn.SiLU()))
                layers.append(nn.Linear(hidden * 2, 1))
                self.accumulator_head = nn.Sequential(*layers)
        if variant == "current_like":
            self.current_consequence = nn.Embedding(feature_vocab + 1, 1, padding_idx=feature_vocab)
        pairs = dense_move_squares()
        if len(pairs) < move_vocab:
            raise ValueError(f"move vocabulary mismatch: {move_vocab} > {len(pairs)}")
        self.register_buffer("move_from", torch.tensor([p[0] for p in pairs[:move_vocab]]))
        self.register_buffer("move_to", torch.tensor([p[1] for p in pairs[:move_vocab]]))
        if variant == "gated":
            self.gate = nn.Linear(hidden, rank)
        elif variant == "hyper2":
            self.move2 = nn.Embedding(move_vocab, rank)
            self.query2 = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, rank))
        elif variant in ("interaction_mlp", "residual_mlp"):
            self.policy_mlp = nn.Sequential(nn.Linear(rank * 3, 64), nn.SiLU(), nn.Linear(64, 1))
        elif variant == "product_mlp":
            self.policy_mlp = nn.Sequential(nn.Linear(rank, 64), nn.SiLU(), nn.Linear(64, 1))
        elif variant.startswith("structured"):
            self.from_square = nn.Embedding(90, rank)
            self.to_square = nn.Embedding(90, rank)
            self.moving_piece = nn.Embedding(15, rank)
            self.captured_piece = nn.Embedding(15, rank)
            self.check = nn.Embedding(2, rank)
            if "consequence" in variant:
                self.consequence = nn.Embedding(feature_vocab + 1, rank, padding_idx=feature_vocab)
                policy_input = rank * 5
            else:
                policy_input = rank * 3
            if "moe" in variant:
                self.policy_moe_gate = nn.Linear(policy_input, 8)
                self.policy_moe_experts = nn.ModuleList(
                    nn.Sequential(nn.Linear(policy_input, rank * 2), nn.SiLU(), nn.Linear(rank * 2, 1))
                    for _ in range(8))
            else:
                self.policy_mlp = nn.Sequential(nn.Linear(policy_input, rank * 2), nn.SiLU(), nn.Linear(rank * 2, 1))
            if "accumulator" in variant:
                if "factor" in variant:
                    self.accumulator_factor_hidden = nn.Linear(hidden, rank, bias=False)
                    self.accumulator_factor_move = nn.Embedding(move_vocab, rank)
                    nn.init.zeros_(self.accumulator_factor_move.weight)
                else:
                    self.accumulator_residual = nn.Sequential(
                        nn.Linear(hidden + rank, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, 1))
                    nn.init.zeros_(self.accumulator_residual[-1].weight)
                    nn.init.zeros_(self.accumulator_residual[-1].bias)
        elif variant not in ("dot", "current_like") and not variant.startswith("full_accumulator"):
            raise ValueError(variant)
        self.value = nn.Sequential(nn.Linear(hidden, 96), nn.SiLU(), nn.Linear(96, 3))

    def initialize_current(self, path):
        with safe_open(path, framework="pt", device="cpu") as weights:
            self.feature.weight.data[:1260].copy_(weights.get_tensor("input_hidden"))
            self.feature.weight.data[1260].zero_()
            self.pretrained_bias = nn.Parameter(weights.get_tensor("hidden_bias").clone())
            self.old_context = nn.Linear(self.feature.embedding_dim, 16, bias=False)
            self.old_context.weight.data.copy_(weights.get_tensor("policy_context_hidden"))
            self.old_move = nn.Embedding(self.move.num_embeddings, 16)
            self.old_move.weight.data.copy_(weights.get_tensor("policy_move_context"))
            self.move_bias.weight.data.copy_(weights.get_tensor("policy_move_bias")[:, None])
            scores = (weights.get_tensor("input_hidden")[:, :32]
                      * weights.get_tensor("policy_consequence_output")[None]).sum(1)
            self.register_buffer("old_consequence", torch.cat((scores, torch.zeros(1))))
        last = self.policy_mlp[-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, features, board, moves, checks, mask):
        tokens = self.feature(features)
        spatial = None
        if hasattr(self, "board_conv"):
            spatial = self.board_piece(board).permute(0, 2, 1).reshape(len(board), -1, 10, 9)
            spatial = spatial + self.board_conv(spatial)
            context = self.trunk(self.norm(self.board_proj(spatial.flatten(1))))
        elif hasattr(self, "board_attention"):
            padding = features == self.feature.padding_idx
            tokens = self.board_attention(tokens, src_key_padding_mask=padding)
            context = (tokens.masked_fill(padding[..., None], 0).sum(1)
                       / (~padding).sum(1, keepdim=True).clamp_min(1))
            context = self.trunk(self.norm(context))
        else:
            summed = tokens.sum(1)
            if hasattr(self, "pretrained_bias"):
                summed = summed + self.pretrained_bias
            context = self.trunk(self.norm(summed))
        if self.variant.startswith("full_accumulator"):
            sources, targets = self.move_from[moves], self.move_to[moves]
            moving = board.gather(1, sources)
            captured = board.gather(1, targets)
            moving_kind = moving - 1
            padding_id = torch.full_like(targets, self.feature.padding_idx)
            after_id = torch.where(moving > 0, moving_kind * 90 + targets, padding_id)
            before_id = torch.where(moving > 0, moving_kind * 90 + sources, padding_id)
            captured_id = torch.where(captured > 0, (captured - 1) * 90 + targets, padding_id)
            root = tokens.sum(1)
            after = root[:, None] + self.feature(after_id) - self.feature(before_id) - self.feature(captured_id)
            move_state = self.accumulator_move(moves) + self.accumulator_check(checks)
            policy_input = torch.cat((F.rms_norm(after, (after.shape[-1],)), move_state), -1)
            if hasattr(self, "moe_gate"):
                gate_logits = self.moe_gate(policy_input)
                top_values, top_indices = gate_logits.topk(2, dim=-1)
                top_weights = top_values.softmax(-1)
                gate_prob = gate_logits.softmax(-1)
                assignment = F.one_hot(top_indices, len(self.moe_experts)).float().sum(-2) * 0.5
                valid = mask[..., None]
                denominator = valid.sum().clamp_min(1)
                self.moe_aux_loss = 8 * (((gate_prob * valid).sum((0, 1)) / denominator)
                                         * ((assignment * valid).sum((0, 1)) / denominator)).sum()
                expert_logits = torch.stack(
                    [expert(policy_input).squeeze(-1) for expert in self.moe_experts], -1)
                logits = ((expert_logits * gate_prob).sum(-1) if self.training else
                          expert_logits.gather(-1, top_indices).mul(top_weights).sum(-1))
            else:
                logits = self.accumulator_head(policy_input).squeeze(-1)
            logits = logits + self.move_bias(moves).squeeze(-1)
            return logits.masked_fill(~mask, -1e4), self.value(context)
        move = self.move(moves)
        if self.variant.startswith("structured"):
            sources, targets = self.move_from[moves], self.move_to[moves]
            move = move + self.from_square(sources) + self.to_square(targets)
            moving = board.gather(1, sources)
            captured = board.gather(1, targets)
            move = move + self.moving_piece(moving) + self.captured_piece(captured)
            move = move + self.check(checks)
            if hasattr(self, "cross_attention"):
                piece_tokens = self.cross_piece(tokens)
                attended, _ = self.cross_attention(move, piece_tokens, piece_tokens,
                                                    key_padding_mask=features == self.feature.padding_idx,
                                                    need_weights=False)
                move = move + attended
            if spatial is not None:
                flat = spatial.flatten(2).transpose(1, 2)
                source_local = flat.gather(1, sources[..., None].expand(-1, -1, flat.shape[-1]))
                target_local = flat.gather(1, targets[..., None].expand(-1, -1, flat.shape[-1]))
                move = move + self.move_local(torch.cat((source_local, target_local), -1))
        query = self.query(context)
        if self.variant in ("dot", "current_like"):
            logits = (move * query[:, None]).sum(-1)
            if self.variant == "current_like":
                sources, targets = self.move_from[moves], self.move_to[moves]
                moving = board.gather(1, sources)
                captured = board.gather(1, targets)
                moving_kind = moving - 1
                padding_id = torch.full_like(targets, self.feature.padding_idx)
                after_id = torch.where(moving > 0, moving_kind * 90 + targets, padding_id)
                before_id = torch.where(moving > 0, moving_kind * 90 + sources, padding_id)
                captured_id = torch.where(captured > 0, (captured - 1) * 90 + targets, padding_id)
                logits += (self.current_consequence(after_id) - self.current_consequence(before_id)
                           - self.current_consequence(captured_id)).squeeze(-1)
        elif self.variant == "gated":
            logits = (move * query[:, None] * (2 * torch.sigmoid(self.gate(context)))[:, None]).sum(-1)
        elif self.variant == "hyper2":
            logits = (move * query[:, None]).sum(-1)
            logits += (self.move2(moves) * self.query2(context)[:, None]).sum(-1)
        elif self.variant in ("interaction_mlp", "residual_mlp") or self.variant.startswith("structured"):
            q = query[:, None].expand_as(move)
            policy_inputs = [q, move, q * move]
            if "consequence" in self.variant:
                moving_kind = moving - 1
                padding_id = torch.full_like(targets, self.feature.padding_idx)
                after_id = torch.where(moving > 0, moving_kind * 90 + targets, padding_id)
                before_id = torch.where(moving > 0, moving_kind * 90 + sources, padding_id)
                captured_id = torch.where(captured > 0, (captured - 1) * 90 + targets,
                                          padding_id)
                delta = self.consequence(after_id) - self.consequence(before_id) - self.consequence(captured_id)
                policy_inputs.extend((delta, q * delta))
            policy_input = torch.cat(policy_inputs, -1)
            if hasattr(self, "piece_moe_experts"):
                expert_logits = torch.stack(
                    [expert(policy_input).squeeze(-1) for expert in self.piece_moe_experts], -1)
                route = torch.where(checks.bool(), torch.full_like(checks, 7), (moving - 1).clamp(0, 6))
                logits = expert_logits.gather(-1, route[..., None]).squeeze(-1)
            elif hasattr(self, "policy_moe_gate"):
                gate_logits = self.policy_moe_gate(policy_input)
                top_values, top_indices = gate_logits.topk(2, dim=-1)
                top_weights = top_values.softmax(-1)
                gate_prob = gate_logits.softmax(-1)
                assignment = F.one_hot(top_indices, len(self.policy_moe_experts)).float().sum(-2) * 0.5
                valid = mask[..., None]
                denominator = valid.sum().clamp_min(1)
                self.moe_aux_loss = 8 * (((gate_prob * valid).sum((0, 1)) / denominator)
                                         * ((assignment * valid).sum((0, 1)) / denominator)).sum()
                expert_logits = torch.stack(
                    [expert(policy_input).squeeze(-1) for expert in self.policy_moe_experts], -1)
                logits = ((expert_logits * gate_prob).sum(-1) if self.training else
                          expert_logits.gather(-1, top_indices).mul(top_weights).sum(-1))
            else:
                logits = self.policy_mlp(policy_input).squeeze(-1)
            if hasattr(self, "accumulator_residual"):
                root = tokens.sum(1)
                after = root[:, None] + self.feature(after_id) - self.feature(before_id) - self.feature(captured_id)
                accumulator_input = torch.cat((F.rms_norm(after, (after.shape[-1],)), move), -1)
                logits += self.accumulator_residual(accumulator_input).squeeze(-1)
            elif hasattr(self, "accumulator_factor_hidden"):
                root = tokens.sum(1)
                after = root[:, None] + self.feature(after_id) - self.feature(before_id) - self.feature(captured_id)
                logits += (self.accumulator_factor_hidden(F.rms_norm(after, (after.shape[-1],)))
                           * self.accumulator_factor_move(moves)).sum(-1)
            if self.variant in ("residual_mlp", "structured_residual"):
                logits += (q * move).sum(-1)
        else:
            logits = self.policy_mlp(query[:, None] * move).squeeze(-1)
        logits = logits + self.move_bias(moves).squeeze(-1)
        if hasattr(self, "old_context"):
            old_logits = (self.old_context(tokens.sum(1) + self.pretrained_bias)[:, None]
                          * self.old_move(moves)).sum(-1)
            moving_kind = moving - 1
            padding_id = torch.full_like(targets, 1260)
            after_id = torch.where(moving > 0, moving_kind * 90 + targets, padding_id)
            before_id = torch.where(moving > 0, moving_kind * 90 + sources, padding_id)
            captured_id = torch.where(captured > 0, (captured - 1) * 90 + targets, padding_id)
            old_logits += self.old_consequence[after_id] - self.old_consequence[before_id] - self.old_consequence[captured_id]
            logits += old_logits
        return logits.masked_fill(~mask, -1e4), self.value(context)


@torch.inference_mode()
def evaluate(model, tensors, indices, batch_size):
    model.eval()
    temperatures = (0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0)
    temperature_ce = [0.0] * len(temperatures)
    total_ce = total_top1 = total_top3 = total_value = 0.0
    features, board, moves, checks, mask, best, wdl = tensors
    for ids in indices.split(batch_size):
        logits, value = model(features[ids], board[ids], moves[ids], checks[ids], mask[ids])
        total_ce += F.cross_entropy(logits.float(), best[ids], reduction="sum").item()
        for index, temperature in enumerate(temperatures):
            temperature_ce[index] += F.cross_entropy(logits.float() / temperature, best[ids], reduction="sum").item()
        order = logits.topk(3, dim=1).indices
        total_top1 += (order[:, 0] == best[ids]).sum().item()
        total_top3 += (order == best[ids, None]).any(1).sum().item()
        total_value += (-(wdl[ids] * F.log_softmax(value.float(), -1)).sum(-1)).sum().item()
    n = len(indices)
    calibrated_index = min(range(len(temperatures)), key=lambda i: temperature_ce[i])
    return {"policy_ce": total_ce / n,
            "calibrated_policy_ce": temperature_ce[calibrated_index] / n,
            "policy_temperature": temperatures[calibrated_index], "top1": total_top1 / n,
            "top3": total_top3 / n, "value_ce": total_value / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--variants", default="dot,gated,hyper2,interaction_mlp")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", default="eval/torch-policy-search")
    parser.add_argument("--init-model")
    parser.add_argument("--value-weight", type=float, default=0.0)
    parser.add_argument("--margin-weight", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--train-groups", type=int, default=0,
                        help="Use a nested prefix of shuffled non-validation groups (0 = all)")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    loaded = load_data(args.dataset)
    *cpu_tensors, feature_vocab, move_vocab = loaded
    tensors = tuple(t.to(device) for t in cpu_tensors)
    model_tensors = tensors[:7]
    count = len(cpu_tensors[0])
    group_count = int(cpu_tensors[7].max()) + 1
    group_order = torch.randperm(group_count, generator=torch.Generator().manual_seed(args.seed)).to(device)
    validation_groups = torch.zeros(group_count, dtype=torch.bool, device=device)
    validation_groups[group_order[-max(1, group_count // 10):]] = True
    is_validation = validation_groups[tensors[7]]
    if args.train_groups:
        available_groups = group_count - max(1, group_count // 10)
        selected_groups = torch.zeros(group_count, dtype=torch.bool, device=device)
        selected_groups[group_order[:min(args.train_groups, available_groups)]] = True
        train_ids = selected_groups[tensors[7]].nonzero().flatten()
    else:
        train_ids = (~is_validation).nonzero().flatten()
    validation_ids = (is_validation & (tensors[8] == 0)).nonzero().flatten()
    split = len(train_ids)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in args.variants.split(","):
        torch.manual_seed(args.seed)
        model = PolicyNet(variant, feature_vocab, move_vocab, args.hidden, args.rank).to(device)
        if args.init_model:
            model.initialize_current(args.init_model)
            model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        model.train()
        steps = processed = 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        for epoch in range(args.epochs):
            epoch_ids = train_ids[torch.randperm(split, device=device)]
            for ids in epoch_ids.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, value = model(*(t[ids] for t in tensors[:5]))
                    policy_logits = logits.float()
                    targets = tensors[5][ids]
                    loss = F.cross_entropy(policy_logits, targets)
                    if args.margin_weight:
                        correct = policy_logits.gather(1, targets[:, None]).squeeze(1)
                        wrong = policy_logits.scatter(1, targets[:, None], -torch.inf).amax(1)
                        loss += args.margin_weight * F.relu(args.margin - correct + wrong).mean()
                    loss += args.value_weight * (-(tensors[6][ids] * F.log_softmax(value.float(), -1)).sum(-1)).mean()
                    if hasattr(model, "moe_aux_loss"):
                        loss += 0.01 * model.moe_aux_loss
                loss.backward(); optimizer.step()
                steps += 1; processed += len(ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = evaluate(model, model_tensors, validation_ids, args.batch_size)
        training_metrics = evaluate(model, model_tensors, train_ids, args.batch_size)
        row = {"variant": variant, "parameters": sum(p.numel() for p in model.parameters()),
               "seconds": elapsed, "steps": steps, "processed": processed,
               "epochs": args.epochs, "samples_per_second": processed / elapsed,
               "train_policy_ce": training_metrics["policy_ce"],
               "train_top1": training_metrics["top1"], **metrics}
        results.append(row)
        torch.save(model.state_dict(), output / f"{variant}.pt")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
