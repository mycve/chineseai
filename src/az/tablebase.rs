//! 与本引擎规则、完整对局历史绑定的离线残局标签。
//!
//! CDB/AXF 原始结果独立存放，不进入此查询路径。`verified_labels` 只能由使用本引擎
//! 规则的证明器写入；其标签仅供训练重标注，调用方不可据此伪造 MCTS 的 solved。
use crate::xiangqi::{Color, Position, RuleHistoryEntry};
use rusqlite::{Connection, OpenFlags, params};
use std::{collections::HashMap, io, path::Path};

/// 修改规则实现、历史字段语义时必须递增；旧规则标签不能跨版本使用。
pub const TABLEBASE_RULESET: &str = "chineseai-rules-v1";
pub const TABLEBASE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS verified_labels (
    ruleset TEXT NOT NULL,
    fen TEXT NOT NULL,
    context BLOB NOT NULL,
    value INTEGER NOT NULL CHECK(value BETWEEN -1 AND 1),
    moves_left INTEGER CHECK(moves_left >= 0 AND moves_left <= 65535),
    verifier TEXT NOT NULL CHECK(length(verifier) > 0),
    PRIMARY KEY(ruleset, fen, context)
) WITHOUT ROWID;
";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TablebaseLabel {
    /// 当前行棋方视角，-1/0/1；不是云库分析分数。
    pub value: i8,
    /// 仅允许规则一致的精确终局距离；DTC 不是该字段的有效来源。
    pub moves_left: Option<u16>,
}

impl TablebaseLabel {
    pub fn wdl(self) -> [f32; 3] {
        match self.value {
            1 => [1.0, 0.0, 0.0],
            0 => [0.0, 1.0, 0.0],
            -1 => [0.0, 0.0, 1.0],
            _ => unreachable!("tablebase labels are validated at insertion and load"),
        }
    }
}

#[derive(Debug, Default)]
pub struct AzTablebase {
    labels: HashMap<(String, Vec<u8>), TablebaseLabel>,
}

impl AzTablebase {
    /// 一次只读加载；自博弈热路径无 SQLite 锁和联网请求。
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(sqlite_error)?;
        let mut query = connection
            .prepare("SELECT fen, context, value, moves_left, verifier FROM verified_labels WHERE ruleset = ?1")
            .map_err(sqlite_error)?;
        let mut rows = query.query([TABLEBASE_RULESET]).map_err(sqlite_error)?;
        let mut labels = HashMap::new();
        while let Some(row) = rows.next().map_err(sqlite_error)? {
            let fen: String = row.get(0).map_err(sqlite_error)?;
            let context: Vec<u8> = row.get(1).map_err(sqlite_error)?;
            let value: i8 = row.get(2).map_err(sqlite_error)?;
            let moves_left: Option<u16> = row.get(3).map_err(sqlite_error)?;
            let verifier: String = row.get(4).map_err(sqlite_error)?;
            if !(-1..=1).contains(&value) || context.is_empty() || verifier.trim().is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid verified tablebase label",
                ));
            }
            let position = Position::from_fen(&fen)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            if position.to_fen() != fen {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "non-canonical tablebase FEN",
                ));
            }
            labels.insert((fen, context), TablebaseLabel { value, moves_left });
        }
        Ok(Self { labels })
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn probe(
        &self,
        position: &Position,
        history: &[RuleHistoryEntry],
    ) -> Option<TablebaseLabel> {
        let context = tablebase_rule_context(position, history)?;
        self.labels.get(&(position.to_fen(), context)).copied()
    }
}

/// 完整历史编码而非散列。包含自然限着、重复开关及所有规则字段，不能仅用 FEN 匹配。
pub fn tablebase_rule_context(
    position: &Position,
    history: &[RuleHistoryEntry],
) -> Option<Vec<u8>> {
    let last = history.last()?;
    if last.hash != position.hash()
        || last.side_to_move != position.side_to_move()
        || last.rule60_clock != position.halfmove_clock()
    {
        return None;
    }
    let color = |side: Color| match side {
        Color::Red => 0u8,
        Color::Black => 1,
    };
    let mut bytes = Vec::with_capacity(16 + history.len() * 35);
    bytes.extend_from_slice(&position.rule60_max_ply().unwrap_or(0).to_le_bytes());
    bytes.push(u8::from(position.repetition_draw_enabled()));
    bytes.extend_from_slice(&position.rule60_count_with_history(history).to_le_bytes());
    bytes.extend_from_slice(&(history.len() as u64).to_le_bytes());
    for entry in history {
        bytes.extend_from_slice(&entry.hash.to_le_bytes());
        bytes.push(color(entry.side_to_move));
        bytes.push(entry.mover.map_or(2, color));
        bytes.push(u8::from(entry.gives_check));
        bytes.extend_from_slice(&entry.chased_mask.to_le_bytes());
        bytes.extend_from_slice(&entry.mv.map_or([255, 255], |mv| [mv.from, mv.to]));
        bytes.push(
            entry
                .captured
                .map_or(0, |piece| 1 + piece.kind as u8 * 2 + color(piece.color)),
        );
        bytes.extend_from_slice(&entry.rule60_clock.to_le_bytes());
    }
    Some(bytes)
}

/// 调用方负责先用本引擎规则完成证明。此函数只负责存储和上下文绑定，不验证胜负。
/// 禁止直接把远端 CDB 查询结果传入此接口。
pub fn write_verified_tablebase_label(
    connection: &Connection,
    position: &Position,
    history: &[RuleHistoryEntry],
    label: TablebaseLabel,
    verifier: &str,
) -> io::Result<()> {
    if !(-1..=1).contains(&label.value) || verifier.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid tablebase label or verifier",
        ));
    }
    let context = tablebase_rule_context(position, history).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "tablebase history does not match position",
        )
    })?;
    connection
        .execute_batch(TABLEBASE_SCHEMA)
        .map_err(sqlite_error)?;
    connection.execute(
        "INSERT INTO verified_labels (ruleset, fen, context, value, moves_left, verifier) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(ruleset, fen, context) DO UPDATE SET value=excluded.value, moves_left=excluded.moves_left, verifier=excluded.verifier",
        params![TABLEBASE_RULESET, position.to_fen(), context, label.value, label.moves_left, verifier],
    ).map_err(sqlite_error)?;
    Ok(())
}

fn sqlite_error(error: rusqlite::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position() -> Position {
        Position::from_fen("4k4/9/9/9/9/9/9/4R4/9/3K5 w - - 0 1").unwrap()
    }

    #[test]
    fn context_changes_for_rules_clocks_and_history() {
        let mut position = position();
        let history = position.initial_rule_history();
        let original = tablebase_rule_context(&position, &history).unwrap();
        position.set_rule60_max_ply(None);
        assert_ne!(
            tablebase_rule_context(&position, &history).unwrap(),
            original
        );
        position.set_rule60_max_ply(Some(120));
        position.set_repetition_draw_enabled(false);
        assert_ne!(
            tablebase_rule_context(&position, &history).unwrap(),
            original
        );
        position.set_repetition_draw_enabled(true);
        let mut other = history.clone();
        other[0].chased_mask ^= 1;
        assert_ne!(tablebase_rule_context(&position, &other).unwrap(), original);
        other[0].rule60_clock += 1;
        assert!(tablebase_rule_context(&position, &other).is_none());
        assert!(tablebase_rule_context(&position, &[]).is_none());
    }

    #[test]
    fn roundtrip_and_rule_isolation() {
        let path =
            std::env::temp_dir().join(format!("chineseai-tablebase-{}.sqlite", std::process::id()));
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch("DROP TABLE IF EXISTS verified_labels;")
            .unwrap();
        let mut position = position();
        let history = position.initial_rule_history();
        let label = TablebaseLabel {
            value: 1,
            moves_left: Some(7),
        };
        write_verified_tablebase_label(
            &connection,
            &position,
            &history,
            label,
            "unit-test-fixture",
        )
        .unwrap();
        // 未核对规则的云库记录和其他版本的本地标签均不会被训练查询加载。
        connection.execute_batch("CREATE TABLE IF NOT EXISTS cdb_labels(fen TEXT); INSERT INTO cdb_labels VALUES ('external');").unwrap();
        connection.execute("INSERT INTO verified_labels SELECT 'other-rules', fen, context, -1, NULL, verifier FROM verified_labels", []).unwrap();
        let database = AzTablebase::load(&path).unwrap();
        assert_eq!(database.len(), 1);
        assert_eq!(database.probe(&position, &history), Some(label));
        position.set_rule60_max_ply(Some(80));
        assert!(database.probe(&position, &history).is_none());
        drop(connection);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn writer_rejects_invalid_labels_and_mismatched_history() {
        let connection = Connection::open_in_memory().unwrap();
        let position = position();
        let history = position.initial_rule_history();
        assert!(
            write_verified_tablebase_label(
                &connection,
                &position,
                &history,
                TablebaseLabel {
                    value: 2,
                    moves_left: None
                },
                "test"
            )
            .is_err()
        );
        assert!(
            write_verified_tablebase_label(
                &connection,
                &position,
                &[],
                TablebaseLabel {
                    value: 1,
                    moves_left: None
                },
                "test"
            )
            .is_err()
        );
    }
}
