#!/usr/bin/env python3
import argparse
import math
import random
from pathlib import Path

INPUT_SIZE = 90 * 14 + 1
V2_INPUT_SIZE = INPUT_SIZE + 2 * 9 * 14 * 90


def load_samples(path: Path):
    samples = []
    max_feature = -1
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            label_text, feature_text = line.split("\t", 1)
        except ValueError as exc:
            raise SystemExit(f"invalid sample format on line {line_no}") from exc
        features = [int(value) for value in feature_text.split()] if feature_text else []
        if features:
            max_feature = max(max_feature, max(features))
        samples.append((float(label_text), features))
    if not samples:
        raise SystemExit("no samples found")
    return samples, max_feature + 1


def init_model(hidden_size: int, input_size: int, init_scale: float):
    scale = init_scale
    input_hidden = [
        [random.uniform(-scale, scale) for _ in range(hidden_size)]
        for _ in range(input_size)
    ]
    hidden_bias = [0.0] * hidden_size
    hidden_output = [random.uniform(-scale, scale) for _ in range(hidden_size)]
    output_bias = 0.0
    return input_hidden, hidden_bias, hidden_output, output_bias


def load_model(path: Path):
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    try:
        input_size = int(values.get("input_size", INPUT_SIZE))
        feature_set = values.get("feature_set", "v1")
        hidden_size = int(values["hidden_size"])
        input_hidden_flat = [float(value) for value in values["input_hidden"].split()]
        hidden_bias = [float(value) for value in values["hidden_bias"].split()]
        hidden_output = [float(value) for value in values["hidden_output"].split()]
        output_bias = float(values["output_bias"])
    except KeyError as exc:
        raise SystemExit(f"missing field in model: {exc}") from exc

    expected = input_size * hidden_size
    if len(input_hidden_flat) != expected:
        raise SystemExit(
            f"input_hidden size mismatch: expected {expected}, got {len(input_hidden_flat)}"
        )
    input_hidden = [
        input_hidden_flat[row * hidden_size : (row + 1) * hidden_size]
        for row in range(input_size)
    ]
    if len(hidden_bias) != hidden_size or len(hidden_output) != hidden_size:
        raise SystemExit("hidden vector size mismatch")
    return input_size, feature_set, hidden_size, input_hidden, hidden_bias, hidden_output, output_bias


def relu(value: float) -> float:
    return value if value > 0.0 else 0.0


def evaluate_loss(samples, input_hidden, hidden_bias, hidden_output, output_bias):
    loss_sum = 0.0
    for target, features in samples:
        hidden = hidden_bias[:]
        for feature in features:
            row = input_hidden[feature]
            for idx in range(len(hidden_bias)):
                hidden[idx] += row[idx]

        activated = [relu(value) for value in hidden]
        output = output_bias
        for idx in range(len(hidden_bias)):
            output += activated[idx] * hidden_output[idx]

        error = output - target
        loss_sum += error * error
    return loss_sum / max(len(samples), 1)


def train(
    samples,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    validation_split: float,
    seed: int,
    resume_model: Path | None,
    input_size: int,
    target_clamp: float,
    grad_clip: float,
    hidden_grad_clip: float,
    init_scale: float,
):
    rng = random.Random(seed)
    random.seed(seed)
    rng.shuffle(samples)
    validation_count = int(len(samples) * validation_split)
    valid_samples = samples[:validation_count]
    train_samples = samples[validation_count:] or samples

    if resume_model is not None:
        loaded_input_size, _, loaded_hidden_size, input_hidden, hidden_bias, hidden_output, output_bias = load_model(
            resume_model
        )
        if loaded_input_size != input_size:
            raise SystemExit(
                f"resume model input size {loaded_input_size} != requested {input_size}"
            )
        if loaded_hidden_size != hidden_size:
            raise SystemExit(
                f"resume model hidden size {loaded_hidden_size} != requested {hidden_size}"
            )
    else:
        input_hidden, hidden_bias, hidden_output, output_bias = init_model(
            hidden_size, input_size, init_scale
        )

    for epoch in range(epochs):
        rng.shuffle(train_samples)
        loss_sum = 0.0
        skipped = 0
        for target, features in train_samples:
            target = clamp(target, -target_clamp, target_clamp)
            hidden = hidden_bias[:]
            for feature in features:
                row = input_hidden[feature]
                for idx in range(hidden_size):
                    hidden[idx] += row[idx]

            activated = [relu(value) for value in hidden]
            output = output_bias
            for idx in range(hidden_size):
                output += activated[idx] * hidden_output[idx]

            error = output - target
            if not math.isfinite(error):
                skipped += 1
                continue
            loss_sum += error * error
            grad = clamp(2.0 * error, -grad_clip, grad_clip)

            output_bias -= learning_rate * grad
            for idx in range(hidden_size):
                old_hidden_output = hidden_output[idx]
                output_grad = grad * activated[idx]
                hidden_output[idx] -= learning_rate * output_grad

                if hidden[idx] <= 0.0:
                    continue
                hidden_grad = clamp(
                    grad * old_hidden_output,
                    -hidden_grad_clip,
                    hidden_grad_clip,
                )
                hidden_bias[idx] -= learning_rate * hidden_grad
                for feature in features:
                    input_hidden[feature][idx] -= learning_rate * hidden_grad

        mean_loss = loss_sum / len(train_samples)
        if valid_samples:
            valid_loss = evaluate_loss(
                valid_samples, input_hidden, hidden_bias, hidden_output, output_bias
            )
            print(
                f"epoch {epoch + 1}: train_mse={mean_loss:.4f} "
                f"train_rmse_cp={math.sqrt(mean_loss):.2f} "
                f"valid_mse={valid_loss:.4f} valid_rmse_cp={math.sqrt(valid_loss):.2f} "
                f"skipped={skipped}"
            )
        else:
            print(
                f"epoch {epoch + 1}: train_mse={mean_loss:.4f} "
                f"train_rmse_cp={math.sqrt(mean_loss):.2f} skipped={skipped}"
            )

    return input_hidden, hidden_bias, hidden_output, output_bias, len(train_samples), len(valid_samples)


def resolve_torch_device(torch, device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_torch_model(
    torch,
    hidden_size: int,
    input_size: int,
    init_scale: float,
    resume_model: Path | None,
    device,
):
    class SparseNnueModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_hidden = torch.nn.EmbeddingBag(
                input_size,
                hidden_size,
                mode="sum",
                include_last_offset=True,
                sparse=False,
            )
            self.hidden_bias = torch.nn.Parameter(torch.zeros(hidden_size))
            self.hidden_output = torch.nn.Parameter(torch.empty(hidden_size))
            self.output_bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, flat_features, offsets):
            hidden = self.input_hidden(flat_features, offsets) + self.hidden_bias
            activated = torch.relu(hidden)
            return activated.matmul(self.hidden_output) + self.output_bias

    if resume_model is not None:
        loaded_input_size, _, loaded_hidden_size, input_hidden, hidden_bias, hidden_output, output_bias = load_model(
            resume_model
        )
        if loaded_input_size != input_size:
            raise SystemExit(
                f"resume model input size {loaded_input_size} != requested {input_size}"
            )
        if loaded_hidden_size != hidden_size:
            raise SystemExit(
                f"resume model hidden size {loaded_hidden_size} != requested {hidden_size}"
            )
    else:
        input_hidden, hidden_bias, hidden_output, output_bias = init_model(
            hidden_size, input_size, init_scale
        )

    model = SparseNnueModel().to(device)
    with torch.no_grad():
        model.input_hidden.weight.copy_(
            torch.tensor(input_hidden, dtype=torch.float32, device=device)
        )
        model.hidden_bias.copy_(torch.tensor(hidden_bias, dtype=torch.float32, device=device))
        model.hidden_output.copy_(torch.tensor(hidden_output, dtype=torch.float32, device=device))
        model.output_bias.copy_(torch.tensor(output_bias, dtype=torch.float32, device=device))
    return model


def torch_batch_tensors(torch, batch, device, target_clamp: float):
    targets = [clamp(target, -target_clamp, target_clamp) for target, _ in batch]
    flat_features = []
    offsets = [0]
    for _, features in batch:
        flat_features.extend(features)
        offsets.append(len(flat_features))
    return (
        torch.tensor(flat_features, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def evaluate_loss_torch(torch, model, samples, batch_size: int, device, target_clamp: float):
    if not samples:
        return 0.0
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            flat_features, offsets, targets = torch_batch_tensors(
                torch, batch, device, target_clamp
            )
            output = model(flat_features, offsets)
            error = output - targets
            loss_sum += float((error * error).sum().detach().cpu())
            count += len(batch)
    model.train()
    return loss_sum / max(count, 1)


def train_torch(
    samples,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    validation_split: float,
    seed: int,
    resume_model: Path | None,
    input_size: int,
    target_clamp: float,
    grad_clip: float,
    init_scale: float,
    batch_size: int,
    device_arg: str,
    optimizer_name: str,
):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install a CUDA build of torch, or use --backend cpu."
        ) from exc

    rng = random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng.shuffle(samples)
    validation_count = int(len(samples) * validation_split)
    valid_samples = samples[:validation_count]
    train_samples = samples[validation_count:] or samples
    device = resolve_torch_device(torch, device_arg)
    model = make_torch_model(
        torch, hidden_size, input_size, init_scale, resume_model, device
    )
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(
        f"torch backend: device={device} batch_size={batch_size} optimizer={optimizer_name}"
    )
    batch_size = max(1, batch_size)
    for epoch in range(epochs):
        rng.shuffle(train_samples)
        model.train()
        loss_sum = 0.0
        skipped = 0
        for start in range(0, len(train_samples), batch_size):
            batch = train_samples[start : start + batch_size]
            flat_features, offsets, targets = torch_batch_tensors(
                torch, batch, device, target_clamp
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(flat_features, offsets)
            loss = torch.mean((output - targets) ** 2)
            if not torch.isfinite(loss):
                skipped += len(batch)
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch)

        mean_loss = loss_sum / len(train_samples)
        if valid_samples:
            valid_loss = evaluate_loss_torch(
                torch, model, valid_samples, batch_size, device, target_clamp
            )
            print(
                f"epoch {epoch + 1}: train_mse={mean_loss:.4f} "
                f"train_rmse_cp={math.sqrt(mean_loss):.2f} "
                f"valid_mse={valid_loss:.4f} valid_rmse_cp={math.sqrt(valid_loss):.2f} "
                f"skipped={skipped}"
            )
        else:
            print(
                f"epoch {epoch + 1}: train_mse={mean_loss:.4f} "
                f"train_rmse_cp={math.sqrt(mean_loss):.2f} skipped={skipped}"
            )

    with torch.no_grad():
        input_hidden = model.input_hidden.weight.detach().cpu().tolist()
        hidden_bias = model.hidden_bias.detach().cpu().tolist()
        hidden_output = model.hidden_output.detach().cpu().tolist()
        output_bias = float(model.output_bias.detach().cpu())
    return input_hidden, hidden_bias, hidden_output, output_bias, len(train_samples), len(valid_samples)


def flatten(matrix):
    return [value for row in matrix for value in row]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def save_model(
    path: Path,
    feature_set: str,
    input_size: int,
    hidden_size: int,
    input_hidden,
    hidden_bias,
    hidden_output,
    output_bias,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"feature_set: {feature_set}",
        f"input_size: {input_size}",
        f"hidden_size: {hidden_size}",
        "input_hidden: " + " ".join(str(value) for value in flatten(input_hidden)),
        "hidden_bias: " + " ".join(str(value) for value in hidden_bias),
        "hidden_output: " + " ".join(str(value) for value in hidden_output),
        f"output_bias: {output_bias}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight NNUE-style model from sparse features.")
    parser.add_argument("input", type=Path, help="Training samples from `cargo run -- nnue-dump`")
    parser.add_argument("output", type=Path, help="Output text model path")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--validation-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--feature-set", choices=["v1", "v2"], default="v1")
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--target-clamp", type=float, default=1200.0)
    parser.add_argument("--grad-clip", type=float, default=64.0)
    parser.add_argument("--hidden-grad-clip", type=float, default=16.0)
    parser.add_argument("--init-scale", type=float, default=0.001)
    parser.add_argument("--backend", choices=["auto", "cpu", "torch"], default="auto")
    parser.add_argument("--device", default="auto", help="torch device: auto, cuda, cuda:0, mps, or cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--torch-optimizer", choices=["adamw", "sgd"], default="adamw")
    args = parser.parse_args()

    samples, min_input_size = load_samples(args.input)
    default_input_size = V2_INPUT_SIZE if args.feature_set == "v2" else INPUT_SIZE
    input_size = args.input_size or default_input_size
    if min_input_size > input_size:
        raise SystemExit(
            f"samples require input size at least {min_input_size}, got {input_size}; "
            "pass --feature-set v2 or --input-size"
        )
    use_torch = args.backend == "torch"
    if args.backend == "auto":
        try:
            import torch

            device = resolve_torch_device(torch, args.device)
            use_torch = device.type != "cpu"
        except Exception:
            use_torch = False

    if use_torch:
        model = train_torch(
            samples,
            args.hidden_size,
            args.epochs,
            args.lr,
            args.validation_split,
            args.seed,
            args.resume,
            input_size,
            args.target_clamp,
            args.grad_clip,
            args.init_scale,
            args.batch_size,
            args.device,
            args.torch_optimizer,
        )
    else:
        if args.backend == "auto":
            print("cpu backend: torch GPU not available")
        model = train(
            samples,
            args.hidden_size,
            args.epochs,
            args.lr,
            args.validation_split,
            args.seed,
            args.resume,
            input_size,
            args.target_clamp,
            args.grad_clip,
            args.hidden_grad_clip,
            args.init_scale,
        )
    input_hidden, hidden_bias, hidden_output, output_bias, train_count, valid_count = model
    save_model(
        args.output,
        args.feature_set,
        input_size,
        args.hidden_size,
        input_hidden,
        hidden_bias,
        hidden_output,
        output_bias,
    )
    print(f"train samples: {train_count}")
    print(f"valid samples: {valid_count}")
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()
