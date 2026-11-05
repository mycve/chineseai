"""下载指定子数以内的完整公开表集。默认仅预览，--download 执行；保留既有表。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ftplib
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

GROUP_SHA256 = "bfa5048fa2293ea3da9ebe9e98dcfefd840229add8a303f12a386b3fe1bc2630"


def connection():
    ftp = ftplib.FTP("ftp.chessdb.cn", timeout=30)
    ftp.login("chessdb", "chessdb")
    return ftp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tablebases/chessdb"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--max-pieces", type=int, default=5)
    parser.add_argument("--limit-gib", type=float, default=1)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.max_pieces <= 13 or not 0 < args.limit_gib:
        raise ValueError("子数或下载上限无效")
    if args.index:
        remote = json.loads(args.index.read_text(encoding="utf-8"))["remote"]
    else:
        remote = {}
        with connection() as ftp:
            for metric in ["DTM", "DTC"]:
                rows = []
                ftp.retrlines(f"LIST /pub/EGTB_{metric}/", rows.append)
                files = []
                for line in rows:
                    fields = line.split(maxsplit=8)
                    if len(fields) == 9 and fields[0].startswith("-"):
                        files.append(dict(name=fields[8], bytes=int(fields[4])))
                remote[metric] = dict(files=files)
    choices = []
    materials = {m: {Path(x["name"]).stem: x for x in remote[m]["files"] if len(Path(x["name"]).stem) <= args.max_pieces} for m in ["DTM", "DTC"]}
    for material in sorted(materials["DTM"].keys() & materials["DTC"].keys()):
        for metric in ["DTM", "DTC"]:
            item = materials[metric][material]
            if Path(item["name"]).name != item["name"]:
                raise ValueError("非法文件名")
            path = args.root / f"EGTB_{metric}" / item["name"]
            if path.exists() and path.stat().st_size != item["bytes"]:
                raise ValueError(f"既有表长度不匹配，停止而非覆盖：{path}")
            choices.append(dict(metric=metric, name=item["name"], bytes=item["bytes"], exists=path.exists()))
    amount = sum(x["bytes"] for x in choices if not x["exists"])
    print(f"materials={len(choices)//2} total_MiB={sum(x['bytes'] for x in choices)/2**20:.2f} download_MiB={amount/2**20:.2f}", flush=True)
    if amount > args.limit_gib * 2**30:
        raise ValueError("超过下载上限；本轮未开始下载")
    if not args.download:
        return
    args.root.mkdir(parents=True, exist_ok=True)
    group = args.root / "GROUP.dat"
    if not group.exists():
        with urlopen("https://raw.githubusercontent.com/noobpwnftw/chessdb/master/extensions/ccegtbprobe/GROUP.dat", timeout=30) as response:
            data = response.read(2 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != GROUP_SHA256:
            raise ValueError("官方 GROUP.dat 版本变化，停止下载")
        with group.open("xb") as output:
            output.write(data)
    if hashlib.sha256(group.read_bytes()).hexdigest() != GROUP_SHA256:
        raise ValueError("既有 GROUP.dat 与索引不匹配，原数据未修改")

    def fetch(item):
        path = args.root / f"EGTB_{item['metric']}" / item["name"]
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            for attempt in range(5):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > item["bytes"]:
                    raise ValueError(f"临时文件大于远端文件：{partial}")
                if offset == item["bytes"]:
                    break
                try:
                    with connection() as ftp, partial.open("ab") as output:
                        ftp.retrbinary(f"RETR /pub/EGTB_{item['metric']}/{item['name']}", output.write, blocksize=262144, rest=offset or None)
                    break
                except (OSError, EOFError, ftplib.Error):
                    if attempt == 4:
                        raise
                    print(f"retry={attempt+1} {item['metric']}/{item['name']}", flush=True)
            if partial.stat().st_size != item["bytes"]:
                raise ValueError(f"下载长度不匹配：{path}")
            # 只创建原先不存在的目标；不覆盖历史表。
            os.link(partial, path)
            partial.unlink()
        return dict(remote=f"/pub/EGTB_{item['metric']}/{item['name']}", bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, choices))
    manifest = args.root / f"download-up-to-{args.max_pieces}-pieces.json"
    manifest.write_text(json.dumps(dict(source="ftp://ftp.chessdb.cn/pub/", files=records), indent=2), encoding="utf-8")
    print(f"verified_files={len(records)} manifest={manifest}")


if __name__ == "__main__":
    main()
