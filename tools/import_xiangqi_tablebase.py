#!/usr/bin/env python3
"""缓存 CDB 外部残局结果；这些记录未经本引擎规则证明，不会直接参与训练。

用法：uv run --no-project tools/import_xiangqi_tablebase.py eval/endgames.fens eval/endgames.sqlite
仅调用 queryall&learn=0；无需第三方 Python 依赖。--self-test 离线测试解析与隔离。
接口：https://chessdb.cn/cloudbook_api.html
规则：https://chessdb.cn/cloudbook_info_en.html （AXF，与本项目规则不能默认视为相同）
"""
import argparse
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

VERIFIED_SCHEMA = """
CREATE TABLE IF NOT EXISTS verified_labels (
    ruleset TEXT NOT NULL, fen TEXT NOT NULL, context BLOB NOT NULL,
    value INTEGER NOT NULL CHECK(value BETWEEN -1 AND 1),
    moves_left INTEGER CHECK(moves_left >= 0 AND moves_left <= 65535),
    verifier TEXT NOT NULL CHECK(length(verifier) > 0),
    PRIMARY KEY(ruleset, fen, context)
) WITHOUT ROWID;
"""
CACHE_SCHEMA = VERIFIED_SCHEMA + """
CREATE TABLE IF NOT EXISTS cdb_responses (
    fen TEXT NOT NULL, metric TEXT NOT NULL CHECK(metric IN ('dtm', 'dtc')),
    response TEXT NOT NULL, status TEXT NOT NULL, fetched_at INTEGER NOT NULL,
    PRIMARY KEY(fen, metric)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS cdb_labels (
    fen TEXT NOT NULL, metric TEXT NOT NULL, move TEXT NOT NULL,
    outcome INTEGER NOT NULL CHECK(outcome BETWEEN -1 AND 1), note TEXT NOT NULL,
    PRIMARY KEY(fen, metric, move)
) WITHOUT ROWID;
"""


def canonical_fen(fen):
    """云库只缓存棋盘和行棋方；绝不冒充完整规则上下文。"""
    fields = fen.split()
    if len(fields) < 2 or fields[1] not in ("w", "b"):
        raise ValueError("FEN 缺少有效行棋方")
    board = fields[0].translate(str.maketrans("heHE", "nbNB"))
    rows = board.split("/")
    if len(rows) != 10:
        raise ValueError("FEN 必须有十行")
    for row in rows:
        width = 0
        for char in row:
            if char in "123456789":
                width += int(char)
            elif char in "kabnrcpKABNRCP":
                width += 1
            else:
                raise ValueError("FEN 含未知棋子")
        if width != 9:
            raise ValueError("FEN 每行必须有九列")
    return f"{board} {fields[1]}"


def parse_cdb_response(response):
    """仅接收文档中带 (W-/D-/L-) 标记的 EGTB 着法，普通 score 永不推断胜负。"""
    labels = []
    for item in response.strip().split("|"):
        fields = dict(part.strip().split(":", 1) for part in item.split(",") if ":" in part)
        move = fields.get("move", "")
        note = fields.get("note", "")
        match = re.search(r"\(\s*([WDL])\s*-([^()]*)\)", note)
        if not re.fullmatch(r"[a-i][0-9][a-i][0-9]", move) or not match:
            continue
        labels.append((move, {"W": 1, "D": 0, "L": -1}[match[1]], note))
    return labels


def store_response(connection, fen, metric, response):
    labels = parse_cdb_response(response)
    status = "egtb" if labels else "unavailable"
    with connection:
        connection.execute("DELETE FROM cdb_labels WHERE fen=? AND metric=?", (fen, metric))
        connection.executemany(
            "INSERT INTO cdb_labels(fen, metric, move, outcome, note) VALUES (?, ?, ?, ?, ?)",
            [(fen, metric, *label) for label in labels],
        )
        connection.execute(
            "INSERT INTO cdb_responses VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(fen, metric) DO UPDATE SET response=excluded.response, "
            "status=excluded.status, fetched_at=excluded.fetched_at",
            (fen, metric, response, status, int(time.time())),
        )
    return len(labels)


def fetch_response(fen, metric, timeout):
    query = urllib.parse.urlencode({"action": "queryall", "board": fen, "egtbmetric": metric, "learn": "0"})
    request = urllib.request.Request("https://www.chessdb.cn/chessdb.php?" + query,
                                     headers={"User-Agent": "chineseai-tablebase-cache/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(1_048_577)
    if len(body) > 1_048_576:
        raise ValueError("云库响应过大")
    return body.decode("utf-8")


def self_test():
    fixture = "move:e2e8,score:29991,rank:2,note:! (W-M-9)|move:d0d1,score:0,rank:1,note:* (D-M-0)"
    assert parse_cdb_response(fixture) == [("e2e8", 1, "! (W-M-9)"), ("d0d1", 0, "* (D-M-0)")]
    assert not parse_cdb_response("move:e2e8,score:30000,rank:2,note:! (44-02)")
    assert not parse_cdb_response("move:e2e8,score:30000,note:W")
    assert not parse_cdb_response("unknown")
    assert not parse_cdb_response("checkmate")
    assert not parse_cdb_response("move:z9z0,score:29991,note:! (W-M-9)")
    fen = canonical_fen("4k4/9/9/9/9/9/9/4R4/9/3K5 w - - 87 1")
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(CACHE_SCHEMA)
        assert store_response(connection, fen, "dtm", fixture) == 2
        assert connection.execute("SELECT count(*) FROM verified_labels").fetchone()[0] == 0
        assert store_response(connection, fen, "dtm", "unknown") == 0
        assert connection.execute("SELECT count(*) FROM cdb_labels").fetchone()[0] == 0
    print("离线测试通过：EGTB 标记解析、普通分数拒绝、未验证缓存隔离。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fens", type=Path, nargs="?", help="每行一个 FEN，# 开头的行忽略")
    parser.add_argument("database", type=Path, nargs="?", help="输出 SQLite 缓存")
    parser.add_argument("--metric", choices=("dtm", "dtc"), default="dtm")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=1.0, help="请求间隔秒数，至少 0.1 秒")
    parser.add_argument("--limit", type=int, default=1000, help="最大联网请求数")
    parser.add_argument("--refresh", action="store_true", help="重新查询已有缓存")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.fens is None or args.database is None:
        parser.error("需要输入 FEN 文件和输出数据库")
    if args.timeout <= 0 or args.interval < 0.1 or args.limit < 1:
        parser.error("timeout>0、interval>=0.1、limit>=1")
    fens = dict.fromkeys(canonical_fen(line) for line in args.fens.read_text(encoding="utf-8").splitlines()
                         if line.strip() and not line.lstrip().startswith("#"))
    args.database.parent.mkdir(parents=True, exist_ok=True)
    queried = labels = cached = errors = 0
    with sqlite3.connect(args.database) as connection:
        connection.executescript(CACHE_SCHEMA)
        for fen in fens:
            if not args.refresh and connection.execute(
                "SELECT 1 FROM cdb_responses WHERE fen=? AND metric=?", (fen, args.metric)
            ).fetchone():
                cached += 1
                continue
            if queried == args.limit:
                break
            if queried:
                time.sleep(args.interval)
            queried += 1
            try:
                labels += store_response(connection, fen, args.metric,
                                         fetch_response(fen, args.metric, args.timeout))
            except (OSError, UnicodeError, ValueError) as error:
                errors += 1
                print(f"查询失败：{error}")
    print(f"请求={queried} 已缓存跳过={cached} 外部着法标签={labels} 失败={errors}；已验证训练标签新增=0")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
