//! 离线 `dive-games` 抽帧结果的 SQLite 存储。
//! 保存局面与抽帧判定依据，供质量抽查与事后筛选。

use std::io;
use std::path::Path;

use rusqlite::Connection;

/// 抽帧库的 schema 版本，写进 `PRAGMA user_version`。
///
/// 抽帧数据是可再生的中间产物（重跑一遍就有），所以版本不匹配时直接**重建空表**，
/// 而不是让 `CREATE TABLE IF NOT EXISTS` 悄悄跳过、留下一个字段对不上的旧表。
/// v1: 只存 fen + 判定依据，去掉 policy/value 标签与全部额外索引。
const DIVE_SCHEMA_VERSION: i32 = 1;

/// 抽帧库的表结构：**只存局面 + 抽帧时的判定依据**。
///
/// 不存 policy / value 目标。`our_q` / `pika_q` / `delta_q`
/// 只用于抽查质量和事后按阈值再筛。
/// 也不加索引：`fen` 的唯一约束已经够用，多一个索引比数据本身还大。
pub fn ensure_dive_schema(conn: &Connection) -> io::Result<()> {
    let stored: i32 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(io_error)?;
    if stored != DIVE_SCHEMA_VERSION {
        conn.execute_batch("DROP TABLE IF EXISTS dives;")
            .map_err(io_error)?;
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

/// 抽帧键只保留棋盘与行棋方，不保留规则计数或重复历史。
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

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub fn io_error(err: rusqlite::Error) -> io::Error {
    io::Error::other(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("dive-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
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
