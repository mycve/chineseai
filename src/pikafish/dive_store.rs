//! 抽帧库（`dive.sqlite`）的 SQLite 存储。
//!
//! 只负责两件事：
//!
//! 1. `dives` 表结构 —— **只有局面本身 + 抽帧时的判定依据**。这些 FEN 是喂回强化
//!    学习、让程序自己探索到劣势的，不产出 policy / value 标签；
//! 2. 按洗牌顺序从 `dives` 抽样作为自博弈起点（[`DiveBook`]）。
//!
//! 开局库**不走这里**：开局局面一律从 `book.pgn.gz` 读，见 `pikafish::opening_book`。

use std::io;
use std::path::Path;

use rusqlite::{Connection, OpenFlags, params};

use crate::az::{AzStartSnapshot, SplitMix64};
use crate::xiangqi::Position;

/// 抽帧库的洗牌盐，避免抽样顺序与其他来源相关。
const DIVE_SHUFFLE_SALT: u64 = 0x0D1F_3B00_4B5E_ED01;

/// 抽帧库的 schema 版本，写进 `PRAGMA user_version`。
///
/// 抽帧数据是可再生的中间产物（重跑一遍就有），所以版本不匹配时直接**重建空表**，
/// 而不是让 `CREATE TABLE IF NOT EXISTS` 悄悄跳过、留下一个字段对不上的旧表。
/// v1: 只存 fen + 判定依据，去掉 policy/value 标签与全部额外索引。
const DIVE_SCHEMA_VERSION: i32 = 1;

/// 抽帧库的表结构：**只存局面 + 抽帧时的判定依据**。
///
/// 不存 policy / value 目标——这些 FEN 是喂给强化学习去探索的，标签由程序自己走
/// 出来。`our_q` / `pika_q` / `delta_q` 只用于抽查质量和事后按阈值再筛。
/// 也不加索引：`fen` 的唯一约束已经够用，多一个索引比数据本身还大。
pub fn ensure_dive_schema(conn: &Connection) -> io::Result<()> {
    let stored: i32 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(io_error)?;
    if stored != DIVE_SCHEMA_VERSION && table_exists(conn, "dives")? {
        conn.execute_batch("DROP TABLE dives;").map_err(io_error)?;
    }
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS dives (
             fen         TEXT    PRIMARY KEY,
             plies       INTEGER NOT NULL,
             our_q       REAL    NOT NULL,
             pika_q      REAL    NOT NULL,
             delta_q     REAL    NOT NULL,
             created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
         );",
    )
    .map_err(io_error)?;
    conn.execute_batch(&format!("PRAGMA user_version = {DIVE_SCHEMA_VERSION};"))
        .map_err(io_error)
}

/// 打开（必要时新建）抽帧库并建表。
pub fn open_dive_db(path: &Path) -> io::Result<Connection> {
    let conn = Connection::open(path).map_err(io_error)?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA temp_store = MEMORY;",
    )
    .map_err(io_error)?;
    ensure_dive_schema(&conn)?;
    Ok(conn)
}

/// FEN 规范化：只保留棋盘与行棋方。
///
/// Px0 book 的 FEN 本身就只有两段，不带 60 回合自然限着计数；旧加载路径用
/// `Position::from_fen` 把计数当 0，所以规范化到两段后行为完全一致。
///
/// **这是有意的信息丢弃，抽帧库的键也只到这一步**：规范化后的 FEN 既不保留
/// `halfmove_clock`（60 回合计数），也不保留任何重复/长将/长捉历史。因此从抽帧库
/// 取出的局面一律是"历史未知"的锚点局面——`rule_context` 里与循环有关的 6 个分量
/// 在这类起点上不可信，只应被当作"全新对局"来用。要真正恢复历史，只能保留着法。
pub fn normalize_fen(fen: &str) -> io::Result<String> {
    let mut parts = fen.split_whitespace();
    let board = parts.next().ok_or_else(|| invalid("empty FEN"))?;
    let side = parts.next().unwrap_or("w");
    match side {
        "w" | "b" | "r" => Ok(format!("{board} {side}")),
        other => Err(invalid(format!("invalid side to move: {other}"))),
    }
}

pub fn side_token(fen: &str) -> &str {
    let mut parts = fen.split_whitespace();
    let _board = parts.next();
    parts.next().unwrap_or("w")
}

/// 只读打开一个已存在的抽帧库。
fn open_readonly(path: &Path) -> io::Result<Connection> {
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .map_err(|err| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("open dive library `{}`: {err}", path.display()),
        )
    })
}

fn table_exists(conn: &Connection, table: &str) -> io::Result<bool> {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = ?1",
            params![table],
            |row| row.get(0),
        )
        .map_err(io_error)?;
    Ok(count > 0)
}

/// 从抽帧库（`dives` 表）按洗牌顺序抽开局局面，用于自博弈。
///
/// 抽到的 FEN 是"我们没意识到自己已经输了"的局面：程序从这些局面出发自博弈，
/// 才能把那个优势/劣势探索出来。这里**只给局面**，不给任何标签。
pub struct DiveBook {
    order: Vec<String>,
    cursor: usize,
}

impl DiveBook {
    pub fn open(path: impl AsRef<Path>, seed: u64) -> io::Result<Self> {
        let conn = open_readonly(path.as_ref())?;
        if !table_exists(&conn, "dives")? {
            return Err(invalid("抽帧库缺少 dives 表：先跑 dive-games"));
        }
        let mut stmt = conn
            .prepare("SELECT fen FROM dives ORDER BY plies, fen")
            .map_err(io_error)?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(io_error)?;
        let mut order = Vec::new();
        for row in rows {
            order.push(row.map_err(io_error)?);
        }
        if order.is_empty() {
            return Err(invalid("抽帧库为空：先跑 dive-games"));
        }
        shuffle(&mut order, seed ^ DIVE_SHUFFLE_SALT);
        Ok(Self { order, cursor: 0 })
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn next_batch(&mut self, count: usize, generation: u32) -> io::Result<Vec<AzStartSnapshot>> {
        let mut snapshots = Vec::with_capacity(count);
        for _ in 0..count {
            if self.cursor == self.order.len() {
                self.cursor = 0;
            }
            let fen = self.order[self.cursor].clone();
            self.cursor += 1;
            snapshots.push(snapshot(&fen, generation)?);
        }
        Ok(snapshots)
    }
}

/// SplitMix64 Fisher-Yates 洗牌，和 `Px0OpeningBook` 用同一套。
fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for index in (1..items.len()).rev() {
        let swap = rng.next_u64() as usize % (index + 1);
        items.swap(index, swap);
    }
}

fn snapshot(fen: &str, generation: u32) -> io::Result<AzStartSnapshot> {
    let position = Position::from_fen(fen).map_err(invalid)?;
    let rule_history = position.initial_rule_history();
    Ok(AzStartSnapshot {
        position,
        rule_history,
        phase_ply: 0,
        generation,
    })
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub fn io_error(err: rusqlite::Error) -> io::Error {
    io::Error::other(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FENS: [&str; 3] = [
        "4k4/9/9/9/9/9/9/9/4P4/4K4 w",
        "4k4/9/9/9/9/9/9/4P4/9/4K4 w",
        "3k5/9/9/9/9/9/9/9/4P4/4K4 b",
    ];

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("dive-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_dive_db(path: &Path) {
        let conn = open_dive_db(path).unwrap();
        for (index, fen) in FENS.iter().enumerate() {
            conn.execute(
                "INSERT INTO dives (fen, plies, our_q, pika_q, delta_q) VALUES (?1, ?2, 0.6, -0.9, 1.5)",
                params![normalize_fen(fen).unwrap(), (14 - index) as i64],
            )
            .unwrap();
        }
    }

    /// 抽帧库只存局面 + 判定依据，不存 policy/value；也不该有额外索引。
    #[test]
    fn dive_schema_is_minimal() {
        let dir = temp_dir("schema");
        let path = dir.join("dive.sqlite");
        let conn = open_dive_db(&path).unwrap();
        let columns: Vec<String> = conn
            .prepare("SELECT name FROM pragma_table_info('dives')")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(
            columns,
            vec!["fen", "plies", "our_q", "pika_q", "delta_q", "created_at"]
        );
        // `PRIMARY KEY` 自带一个 sqlite_autoindex，这是 SQLite 的实现细节；
        // 这里要保证的是**没有我们手工加的额外索引**（多一个索引比数据本身还大）。
        let extra_indexes: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                  WHERE type='index' AND tbl_name='dives' AND name NOT LIKE 'sqlite_autoindex%'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(extra_indexes, 0, "抽帧库不该有额外索引");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dive_book_reads_shuffled_positions() {
        let dir = temp_dir("read");
        let path = dir.join("dive.sqlite");
        write_dive_db(&path);
        let mut book = DiveBook::open(&path, 7).unwrap();
        assert_eq!(book.len(), FENS.len());
        let batch = book.next_batch(3, 5).unwrap();
        assert_eq!(batch.len(), 3);
        assert!(batch.iter().all(|s| s.phase_ply == 0 && s.generation == 5));
        assert_eq!(
            batch
                .iter()
                .map(|s| s.position.hash())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );
        // 洗牌可复现。
        let mut again = DiveBook::open(&path, 7).unwrap();
        let repeated = again.next_batch(3, 5).unwrap();
        for (left, right) in batch.iter().zip(&repeated) {
            assert_eq!(left.position.hash(), right.position.hash());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 新的一局重来一遍（游标回绕）。
    #[test]
    fn dive_book_wraps_around() {
        let dir = temp_dir("wrap");
        let path = dir.join("dive.sqlite");
        write_dive_db(&path);
        let mut book = DiveBook::open(&path, 11).unwrap();
        let first = book.next_batch(5, 0).unwrap();
        assert_eq!(first.len(), 5);
        // 3 条局面、取 5 个：应当回绕复用。
        let hashes = first
            .iter()
            .map(|s| s.position.hash())
            .collect::<Vec<_>>();
        assert_eq!(hashes.len(), 5);
        assert_eq!(hashes[0], hashes[3]);
        assert_eq!(hashes[1], hashes[4]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_or_empty_dive_db_is_an_error() {
        let dir = temp_dir("empty");
        let missing = dir.join("nope.sqlite");
        assert!(DiveBook::open(&missing, 1).is_err());

        let empty = dir.join("empty.sqlite");
        let _ = open_dive_db(&empty).unwrap();
        let err = match DiveBook::open(&empty, 1) {
            Ok(_) => panic!("空库不该打开成功"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("dive-games"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 旧 schema（字段对不上）应当被自动重建为空表，而不是留下坏表。
    #[test]
    fn stale_schema_is_rebuilt() {
        let dir = temp_dir("stale");
        let path = dir.join("dive.sqlite");
        {
            // 造一个"上一版"的表：多出来的列 + 我们已删掉的索引。
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE dives (fen TEXT PRIMARY KEY, zobrist INTEGER NOT NULL, our_q REAL);
                 CREATE INDEX idx_dives_zobrist ON dives(zobrist);",
            )
            .unwrap();
            conn.execute("INSERT INTO dives VALUES ('x', 1, 0.0)", [])
                .unwrap();
        }
        let conn = open_dive_db(&path).unwrap();
        let columns: Vec<String> = conn
            .prepare("SELECT name FROM pragma_table_info('dives')")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(
            columns,
            vec!["fen", "plies", "our_q", "pika_q", "delta_q", "created_at"],
            "旧表应当被重建"
        );
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM dives", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 0, "重建后不该残留旧行");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn normalize_fen_keeps_only_board_and_side() {
        assert_eq!(
            normalize_fen("4k4/9/9/9/9/9/9/9/4P4/4K4 w - - 7 42").unwrap(),
            "4k4/9/9/9/9/9/9/9/4P4/4K4 w"
        );
        assert_eq!(
            normalize_fen("4k4/9/9/9/9/9/9/9/4P4/4K4")
                .unwrap()
                .split_whitespace()
                .count(),
            2
        );
        assert!(normalize_fen("4k4/9/9/9/9/9/9/9/4P4/4K4 x").is_err());
    }

    #[test]
    fn side_token_reads_the_second_field() {
        assert_eq!(side_token("4k4/9/9/9/9/9/9/9/4P4/4K4 w"), "w");
        assert_eq!(side_token("4k4/9/9/9/9/9/9/9/4P4/4K4 b"), "b");
        assert_eq!(side_token("4k4/9/9/9/9/9/9/9/4P4/4K4"), "w");
    }
}
