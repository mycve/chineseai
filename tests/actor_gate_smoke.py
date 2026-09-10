"""使用已编译的 fast 主程序验证 actor 发布、拒绝及重启行为。"""

import hashlib
import pathlib
import subprocess
import sys
import tempfile


CONFIG = '''format_version = 23
simulations = 8
hidden_size = 8
workers = 2
selfplay_samples_per_update = 64
train_samples_per_update = 64
train_warmup_samples = 64
replay_capacity = 512
batch_size = 32
max_plies = 8
opening_start_fraction = 0.0
midgame_start_fraction = 0.0
actor_publish_interval_updates = 1
actor_noninferiority_margin = 0.02
arena_interval = 1
arena_simulations = 8
arena_promotion_rate = 1.0
arena_processes = 2
arena_opening_positions = 0
arena_random_positions = 2
arena_opening_book = ""
pikafish_label_eval_interval = 0
checkpoint_interval = 1
'''


def run(executable, directory, target):
    result = subprocess.run(
        [str(executable), 'az-loop', 'loop.toml', '--target-update', str(target)],
        cwd=directory, text=True, encoding='utf-8', stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=90,
    )
    assert result.returncode == 0, result.stdout
    assert 'promoted=current' not in result.stdout, result.stdout
    return result.stdout


def main():
    executable = pathlib.Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='chineseai-actor-gate-') as temporary:
        directory = pathlib.Path(temporary)
        (directory / 'loop.toml').write_text(CONFIG + 'actor_gate_min_games = 4\n', encoding='utf-8')
        log = run(executable, directory, 2)
        assert 'published learner update 1' in log, log
        assert 'published learner update 2' in log, log
        assert 'actor-match 2:' in log, log
        best_hash = hashlib.sha256((directory / 'best.safetensors').read_bytes()).digest()
        log = run(executable, directory, 3)
        assert 'actor starts from champion' in log, log
        assert 'actor-gate 3: actor_update=0' in log, log
        assert best_hash == hashlib.sha256((directory / 'best.safetensors').read_bytes()).digest()
    with tempfile.TemporaryDirectory(prefix='chineseai-actor-hold-') as temporary:
        directory = pathlib.Path(temporary)
        (directory / 'loop.toml').write_text(CONFIG + 'actor_gate_min_games = 400\n', encoding='utf-8')
        log = run(executable, directory, 2)
        assert 'decision=Hold' in log, log
        assert 'published learner' not in log, log
    print('actor gate smoke: publish without promotion, current-actor match, safe restart, and insufficient-evidence hold passed')


if __name__ == '__main__':
    main()
