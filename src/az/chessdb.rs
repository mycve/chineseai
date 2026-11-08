use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use rusqlite::{Connection, OpenFlags, OptionalExtension};

use crate::xiangqi::{Color, Move, Position};

pub(super) struct SelfplayTablebase(Connection);

const DTM_MIRROR: &str = "rsync://ftp.chessdb.cn/ftp/pub/EGTB_DTM";

pub(super) struct DtmTablebase {
    directory: PathBuf,
    available: HashSet<String>,
    queue: mpsc::SyncSender<String>,
    last_request: Mutex<HashMap<String, Instant>>,
}

impl DtmTablebase {
    pub(super) fn shared() -> Option<&'static Self> {
        static INSTANCE: OnceLock<Option<DtmTablebase>> = OnceLock::new();
        INSTANCE
            .get_or_init(|| Self::open(PathBuf::from("data/chessdb-dtm")))
            .as_ref()
    }

    fn open(directory: PathBuf) -> Option<Self> {
        let index = std::fs::read_to_string(directory.join("files.txt")).ok()?;
        let available = index.lines().map(str::to_owned).collect();
        let (queue, requests) = mpsc::sync_channel::<String>(16);
        let worker_directory = directory.clone();
        std::thread::Builder::new()
            .name("selfplay-dtm-download".into())
            .spawn(move || {
                for name in requests {
                    if !worker_directory.join(&name).is_file() {
                        download_dtm(&worker_directory, &name);
                    }
                }
            })
            .ok()?;
        Some(Self {
            directory,
            available,
            queue,
            last_request: Mutex::new(HashMap::new()),
        })
    }

    pub(super) fn cached_or_request(&self, fen: &str) -> bool {
        let Some(name) = material_files(fen)
            .into_iter()
            .find(|name| self.available.contains(name))
        else {
            return false;
        };
        if self.directory.join(&name).is_file() {
            return true;
        }
        let mut last_request = self.last_request.lock().unwrap();
        let now = Instant::now();
        if last_request
            .get(&name)
            .is_none_or(|last| now.duration_since(*last) >= Duration::from_secs(300))
            && self.queue.try_send(name.clone()).is_ok()
        {
            last_request.insert(name, now);
        }
        false
    }

    pub(super) fn adjudicate(&self, fen: &str, position: &Position) -> Option<f32> {
        if !self.directory.join("probe").is_file() {
            return None;
        }
        let output = local_command("./probe")
            .arg(fen)
            .current_dir(&self.directory)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let score = std::str::from_utf8(&output.stdout)
            .ok()?
            .trim()
            .parse()
            .ok()?;
        dtm_outcome(score, position)
    }
}

fn dtm_outcome(score: i32, position: &Position) -> Option<f32> {
    if score == 0 {
        return Some(0.0);
    }
    // ChessDB 的根节点分数为 ±(30000 - DTM)，DTM 以半回合计。
    let distance = 30_000i32.checked_sub(score.checked_abs()?)?;
    if !(1..=2047).contains(&distance)
        || position
            .rule60_max_ply()
            .is_some_and(|limit| distance >= i32::from(limit))
    {
        return None;
    }
    Some(red_outcome(score.signum() as i8, position))
}

fn material_files(fen: &str) -> [String; 2] {
    let board = fen.split_whitespace().next().unwrap_or_default().as_bytes();
    let mut red = String::new();
    let mut black = String::new();
    for piece in b"KABRNCP" {
        red.extend(
            board
                .iter()
                .filter(|&square| square == piece)
                .map(|&p| p as char),
        );
        black.extend(
            board
                .iter()
                .filter(|&&square| square == piece.to_ascii_lowercase())
                .map(|_| *piece as char),
        );
    }
    [format!("{red}{black}.lzdtm"), format!("{black}{red}.lzdtm")]
}

fn local_command(program: &str) -> Command {
    #[cfg(windows)]
    {
        let mut command = Command::new("wsl.exe");
        command.args(["--exec", program]);
        command
    }
    #[cfg(not(windows))]
    {
        Command::new(program)
    }
}

fn download_dtm(directory: &std::path::Path, name: &str) {
    // rsync 默认在目标目录内写临时文件，传输成功后才重命名为最终文件。
    let _ = local_command("rsync")
        .args(["--ignore-existing", "--timeout=20", "--contimeout=10"])
        .arg(format!("{DTM_MIRROR}/{name}"))
        .arg(".")
        .current_dir(directory)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

pub(super) fn endgame_fen(position: &Position) -> Option<String> {
    let fen = position.to_fen();
    let mut fields = fen.split_whitespace();
    let board = fields.next()?;
    let side = fields.next()?;
    // 限制首次接入的查询规模。帅将也计入棋子数。
    (board.bytes().filter(u8::is_ascii_alphabetic).count() <= 7).then(|| format!("{board} {side}"))
}

impl SelfplayTablebase {
    pub(super) fn open() -> Option<Self> {
        let path = std::env::var_os("CHINESEAI_SELFPLAY_CHESSDB")?;
        match Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY) {
            Ok(db) => Some(Self(db)),
            Err(err) => {
                eprintln!("自博弈残局库打开失败：{err}");
                None
            }
        }
    }

    #[cfg(test)]
    fn open_at(path: impl AsRef<std::path::Path>) -> rusqlite::Result<Self> {
        let db = Connection::open(path)?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS chessdb_dtm (
                fen TEXT PRIMARY KEY,
                response TEXT NOT NULL
            )",
        )?;
        Ok(Self(db))
    }

    pub(super) fn adjudicate(&self, fen: &str, position: &Position, legal: &[Move]) -> Option<f32> {
        let response = self
            .0
            .query_row(
                "SELECT response FROM chessdb_dtm WHERE fen = ?1",
                [fen],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .ok()??;
        let (outcome, distance) = parse_dtm(&response, legal)?;

        // 使用保守上界，为服务端距离单位和首着留出余量。
        if outcome != 0
            && position
                .rule60_max_ply()
                .is_some_and(|limit| distance.saturating_mul(2).saturating_add(2) >= limit)
        {
            return None;
        }
        Some(red_outcome(outcome, position))
    }
}

fn red_outcome(outcome: i8, position: &Position) -> f32 {
    f32::from(match position.side_to_move() {
        Color::Red => outcome,
        Color::Black => -outcome,
    })
}

fn parse_dtm(response: &str, legal: &[Move]) -> Option<(i8, u16)> {
    let mut remaining: Vec<_> = legal.iter().map(|mv| mv.to_uci()).collect();
    let mut best = None;
    for entry in response
        .trim_matches(|c: char| c.is_whitespace() || c == '\0')
        .split('|')
    {
        let mut fields = entry.split(',');
        let mv = fields.next()?.strip_prefix("move:")?;
        let score = fields.next()?.strip_prefix("score:")?.parse::<i32>().ok()?;
        let note = fields.find_map(|field| field.strip_prefix("note:"))?;
        let (_, note) = note.split_once(" (")?;
        let (outcome, distance) = note.strip_suffix(')')?.split_once("-M-")?;
        let outcome = match (outcome, score.signum()) {
            ("W", 1) => 1,
            ("D", 0) => 0,
            ("L", -1) => -1,
            _ => return None,
        };
        let distance = distance.parse::<u16>().ok()?;
        remaining.swap_remove(remaining.iter().position(|candidate| candidate == mv)?);
        if best.is_none_or(|(best_outcome, best_distance)| {
            outcome > best_outcome
                || (outcome == 1 && best_outcome == 1 && distance < best_distance)
                || (outcome == -1 && best_outcome == -1 && distance > best_distance)
        }) {
            best = Some((outcome, distance));
        }
    }
    let best = best?;
    // 一手已证胜的合法着法足以判胜。判和、判负则需要覆盖所有合法着法。
    (best.0 == 1 || remaining.is_empty()).then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_lookup_handles_piece_order_and_swapped_sides() {
        let files = material_files("4ka3/9/6b2/9/9/9/9/9/9/3RK4 b");
        assert_eq!(files, ["KRKAB.lzdtm", "KABKR.lzdtm"]);
    }

    #[test]
    fn dtm_outcome_uses_red_perspective_and_respects_move_limit() {
        let mut position = Position::from_fen("9/9/4k4/9/9/9/8P/9/9/5K3 b").unwrap();
        position.set_rule60_max_ply(Some(20));
        assert_eq!(dtm_outcome(-29981, &position), Some(1.0));
        assert_eq!(dtm_outcome(29981, &position), Some(-1.0));
        assert_eq!(dtm_outcome(0, &position), Some(0.0));
        assert_eq!(dtm_outcome(-29980, &position), None);
        assert_eq!(dtm_outcome(42, &position), None);
    }

    #[test]
    fn sqlite_hit_adjudicates_only_complete_dtm_results() {
        let position = Position::from_fen("9/9/4k4/9/9/9/8P/9/9/5K3 w").unwrap();
        let fen = endgame_fen(&position).unwrap();
        let legal = position.legal_moves();
        let response = legal
            .iter()
            .map(|mv| format!("move:{},score:0,rank:2,note:! (D-M-0000)", mv.to_uci()))
            .collect::<Vec<_>>()
            .join("|");
        let tablebase = SelfplayTablebase::open_at(":memory:").unwrap();
        tablebase
            .0
            .execute("INSERT INTO chessdb_dtm VALUES (?1, ?2)", (&fen, &response))
            .unwrap();
        assert_eq!(tablebase.adjudicate(&fen, &position, &legal), Some(0.0));
        assert_eq!(parse_dtm(response.split('|').next().unwrap(), &legal), None);
        assert_eq!(parse_dtm(&response.replace("D-M-", "D-00-"), &legal), None);
        assert_eq!(
            parse_dtm(
                "move:f0f1,score:29981,rank:2,note:! (W-M-0019)|move:i3i4,score:29981,rank:2,note:! (W-M-0019)\0",
                &legal,
            ),
            Some((1, 19))
        );
    }
}
