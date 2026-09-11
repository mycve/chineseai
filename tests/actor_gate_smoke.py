"""使用已编译的 fast 主程序验证仅晋级发布、保留 best 及重启行为。"""

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
    return result.stdout


def main():
    executable = pathlib.Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='chineseai-promotion-only-') as temporary:
        directory = pathlib.Path(temporary)
        (directory / 'loop.toml').write_text(CONFIG, encoding='utf-8')
        log = run(executable, directory, 2)
        assert 'promoted=current' not in log and 'published champion' not in log, log
        best_hash = hashlib.sha256((directory / 'best.safetensors').read_bytes()).digest()
        log = run(executable, directory, 3)
        assert 'starts from best; publish only after promotion' in log, log
        assert 'published champion' not in log, log
        assert best_hash == hashlib.sha256((directory / 'best.safetensors').read_bytes()).digest()
        config = (directory / 'loop.toml').read_text(encoding='utf-8')
        (directory / 'loop.toml').write_text(config.replace('arena_interval = 1', 'arena_interval = 0'), encoding='utf-8')
        log = run(executable, directory, 4)
        assert 'arena disabled; best remains fixed' in log, log
        assert 'published champion' not in log, log
        assert best_hash == hashlib.sha256((directory / 'best.safetensors').read_bytes()).digest()
    with tempfile.TemporaryDirectory(prefix='chineseai-promotion-publish-') as temporary:
        directory = pathlib.Path(temporary)
        (directory / 'loop.toml').write_text(CONFIG.replace('arena_promotion_rate = 1.0', 'arena_promotion_rate = 0.0'), encoding='utf-8')
        log = run(executable, directory, 2)
        assert 'promoted=current' in log, log
        assert 'published champion update 1' in log, log
        assert 'published champion update 2' in log, log
    print('promotion-only smoke: hold, restart from best, disabled arena, and champion publication passed')


if __name__ == '__main__':
    main()
