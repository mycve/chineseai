"""将格式 33 的模型及可选 SGD 状态显式转换为格式 34；原文件保持不变。"""
import argparse
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

OLD_SIZE = 2062 * 7 * 64 + 7 * 64
EXTRA_SIZE = 7 * 7 * 32
MODEL_WEIGHTS = (
    "input_hidden", "input_piece_hidden", "input_rank_hidden", "input_file_hidden",
    "input_king_piece_hidden", "rule_context_hidden", "hidden_bias", "value_head_hidden",
    "value_head_bias", "value_head_output", "short_value_head_output", "short_value_head_bias",
    "value_threat_embedding", "value_threat_output", "policy_threat_context", "policy_move_bias",
    "policy_consequence_output", "policy_context_hidden", "policy_move_context",
    "policy_accumulator_hidden", "policy_accumulator_move", "policy_sparse_table",
    "policy_sparse_factor", "policy_tactical", "policy_repetition_hidden", "policy_repetition_bias",
)


def extend(array):
    if array.shape != (OLD_SIZE,) or array.dtype != np.float32:
        raise ValueError("原战术张量形状或类型不正确")
    return np.concatenate((array, np.zeros(EXTRA_SIZE, dtype=np.float32)))


def read(path):
    with safe_open(str(path), framework="numpy") as source:
        metadata = source.metadata()
    return load_file(str(path)), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--optimizer", type=Path)
    parser.add_argument("--optimizer-output", type=Path)
    args = parser.parse_args()
    if bool(args.optimizer) != bool(args.optimizer_output):
        parser.error("优化器输入与输出必须同时指定")
    outputs = [args.output] + ([args.optimizer_output] if args.optimizer else [])
    if len({p.resolve() for p in outputs}) != len(outputs) or any(p.exists() for p in outputs):
        parser.error("输出必须是互不相同且尚不存在的文件")
    model, metadata = read(args.model)
    if model["az_model_format_version"].tolist() != [33.0]:
        raise ValueError("仅接受格式 33 模型")
    old_tactical = model["policy_tactical"]
    model["policy_tactical"] = extend(old_tactical)
    model["az_model_format_version"] = np.array([34.0], dtype=np.float32)
    optimizer = None
    if args.optimizer:
        optimizer, optimizer_metadata = read(args.optimizer)
        state = optimizer["state"]
        if state.shape != (4,) or state[0] != 1 or state[1] != 33:
            raise ValueError("仅接受格式 33 的 SGD 状态")
        keys = {k for k in optimizer if k.startswith("weight_")}
        if keys != {f"weight_{i}" for i in range(len(MODEL_WEIGHTS))}:
            raise ValueError("旧 SGD 张量布局不正确")
        for i, name in enumerate(MODEL_WEIGHTS):
            expected = old_tactical if name == "policy_tactical" else model[name]
            weight, velocity = optimizer[f"weight_{i}"], optimizer[f"velocity_{i}"]
            if not np.array_equal(weight, expected) or weight.dtype != expected.dtype:
                raise ValueError(f"SGD 权重与模型不匹配：{name}")
            if velocity.shape != expected.shape or velocity.dtype != expected.dtype:
                raise ValueError(f"SGD 动量布局不正确：{name}")
        index = MODEL_WEIGHTS.index("policy_tactical")
        optimizer[f"weight_{index}"] = extend(old_tactical)
        optimizer[f"velocity_{index}"] = extend(optimizer[f"velocity_{index}"])
        state[1] = 34

    save_file(model, str(args.output), metadata=metadata)
    if optimizer is not None:
        save_file(optimizer, str(args.optimizer_output), metadata=optimizer_metadata)
    print("转换完成；旧权重、SGD 步数和动量保留，新关系权重与动量为零。")


if __name__ == "__main__":
    main()
