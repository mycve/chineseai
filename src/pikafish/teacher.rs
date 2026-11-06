//! 有超时的持久 UCI 教师。只返回本引擎允许的着法，不把教师分数当成终局证明。
use crate::xiangqi::{Move, Position};
use std::{
    collections::HashSet,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, JoinHandle},
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct TeacherConfig {
    pub exe: PathBuf,
    pub nnue: PathBuf,
    pub hash_mb: u32,
    pub timeout: Duration,
}

#[derive(Clone, Copy, Debug)]
pub enum TeacherBudget {
    Depth(u32),
    Nodes(u64),
}

#[derive(Clone, Copy)]
enum ResponseKind {
    Uci,
    Ready,
    BestMove,
}

struct Response {
    line: String,
    options: HashSet<String>,
}

struct Request {
    commands: Vec<String>,
    kind: ResponseKind,
    reply: Sender<io::Result<Response>>,
}

/// 单个实例必须整局独占；多个 actor 可使用固定大小的教师实例池。
#[derive(Debug)]
pub struct PikafishTeacher {
    child: Child,
    requests: Option<Sender<Request>>,
    worker: Option<JoinHandle<()>>,
    timeout: Duration,
}

impl PikafishTeacher {
    pub fn spawn(config: &TeacherConfig) -> io::Result<Self> {
        if config.hash_mb == 0 || config.timeout.is_zero() {
            return Err(invalid("teacher Hash and timeout must be positive"));
        }
        let exe = std::path::absolute(&config.exe)?;
        let nnue = std::path::absolute(&config.nnue)?;
        if !exe.is_file() || !nnue.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "teacher executable or NNUE file missing",
            ));
        }
        let nnue = nnue
            .to_str()
            .ok_or_else(|| invalid("teacher NNUE path is not UTF-8"))?;
        if nnue.contains(['\r', '\n']) {
            return Err(invalid("teacher NNUE path contains a newline"));
        }
        Self::launch(Command::new(exe), nnue, config.hash_mb, config.timeout)
    }

    fn launch(
        mut command: Command,
        nnue: &str,
        hash_mb: u32,
        timeout: Duration,
    ) -> io::Result<Self> {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = command.spawn()?;
        let stdin = child.stdin.take().expect("piped teacher stdin");
        let stdout = child.stdout.take().expect("piped teacher stdout");
        let (sender, receiver) = mpsc::channel::<Request>();
        let worker = match thread::Builder::new()
            .name("pikafish-teacher-io".into())
            .spawn(move || {
                let mut input = BufWriter::new(stdin);
                let mut output = BufReader::new(stdout);
                for request in receiver {
                    let result = transact(&mut input, &mut output, &request.commands, request.kind);
                    let failed = result.is_err();
                    if request.reply.send(result).is_err() || failed {
                        break;
                    }
                }
            }) {
            Ok(worker) => worker,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        let mut teacher = Self {
            child,
            requests: Some(sender),
            worker: Some(worker),
            timeout,
        };
        let response = teacher.exchange(vec!["uci".into()], ResponseKind::Uci)?;
        for option in ["Threads", "Hash", "EvalFile"] {
            if !response.options.contains(option) {
                return Err(invalid(format!(
                    "teacher did not advertise UCI option {option}"
                )));
            }
        }
        teacher.exchange(
            vec![
                "setoption name Threads value 1".into(),
                format!("setoption name Hash value {hash_mb}"),
                format!("setoption name EvalFile value {nnue}"),
                "isready".into(),
            ],
            ResponseKind::Ready,
        )?;
        Ok(teacher)
    }

    pub fn new_game(&mut self) -> io::Result<()> {
        self.exchange(
            vec!["ucinewgame".into(), "isready".into()],
            ResponseKind::Ready,
        )?;
        Ok(())
    }

    pub fn best_move(
        &mut self,
        position_command: &str,
        legal_moves: &[Move],
        budget: TeacherBudget,
    ) -> io::Result<Move> {
        validate_position_command(position_command)?;
        if legal_moves.is_empty() {
            return Err(invalid("teacher received no legal moves"));
        }
        let limit = match budget {
            TeacherBudget::Depth(0) | TeacherBudget::Nodes(0) => {
                return Err(invalid("teacher search budget must be positive"));
            }
            TeacherBudget::Depth(depth) => format!("depth {depth}"),
            TeacherBudget::Nodes(nodes) => format!("nodes {nodes}"),
        };
        let allowed = legal_moves
            .iter()
            .map(|mv| mv.to_uci())
            .collect::<Vec<_>>()
            .join(" ");
        let response = self.exchange(
            vec![
                position_command.into(),
                format!("go {limit} searchmoves {allowed}"),
            ],
            ResponseKind::BestMove,
        )?;
        match parse_best_move(&response.line, legal_moves) {
            Ok(mv) => Ok(mv),
            Err(error) => {
                self.terminate();
                Err(error)
            }
        }
    }

    fn exchange(&mut self, commands: Vec<String>, kind: ResponseKind) -> io::Result<Response> {
        let sender = self.requests.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "teacher process is closed")
        })?;
        let (reply, receiver) = mpsc::channel();
        if sender
            .send(Request {
                commands,
                kind,
                reply,
            })
            .is_err()
        {
            self.terminate();
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "teacher I/O worker stopped",
            ));
        }
        let result = receive_response(&receiver, self.timeout);
        if result.is_err() {
            self.terminate();
        }
        result
    }

    fn terminate(&mut self) {
        self.requests.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for PikafishTeacher {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn receive_response(
    receiver: &Receiver<io::Result<Response>>,
    timeout: Duration,
) -> io::Result<Response> {
    receiver
        .recv_timeout(timeout)
        .map_err(|error| match error {
            mpsc::RecvTimeoutError::Timeout => io::Error::new(
                io::ErrorKind::TimedOut,
                "Pikafish teacher request timed out",
            ),
            mpsc::RecvTimeoutError::Disconnected => io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Pikafish teacher output closed",
            ),
        })?
}

fn transact(
    input: &mut impl Write,
    output: &mut impl BufRead,
    commands: &[String],
    kind: ResponseKind,
) -> io::Result<Response> {
    for command in commands {
        writeln!(input, "{command}")?;
    }
    input.flush()?;
    let mut options = HashSet::new();
    let mut line = String::new();
    loop {
        line.clear();
        if output.read_line(&mut line)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Pikafish teacher exited before its UCI response",
            ));
        }
        let trimmed = line.trim();
        if trimmed.starts_with("info string ERROR") || trimmed.starts_with("ERROR:") {
            return Err(io::Error::other(format!("Pikafish teacher: {trimmed}")));
        }
        if matches!(kind, ResponseKind::Uci) {
            if let Some((name, _)) = trimmed
                .strip_prefix("option name ")
                .and_then(|text| text.split_once(" type "))
            {
                options.insert(name.to_owned());
            }
        }
        let done = match kind {
            ResponseKind::Uci => trimmed == "uciok",
            ResponseKind::Ready => trimmed == "readyok",
            ResponseKind::BestMove => trimmed.split_whitespace().next() == Some("bestmove"),
        };
        if done {
            return Ok(Response {
                line: trimmed.to_owned(),
                options,
            });
        }
    }
}

fn valid_move_text(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 4
        && (b'a'..=b'i').contains(&bytes[0])
        && bytes[1].is_ascii_digit()
        && (b'a'..=b'i').contains(&bytes[2])
        && bytes[3].is_ascii_digit()
}

fn parse_best_move(line: &str, legal_moves: &[Move]) -> io::Result<Move> {
    let mut words = line.split_whitespace();
    if words.next() != Some("bestmove") {
        return Err(invalid("missing teacher bestmove"));
    }
    let text = words
        .next()
        .filter(|text| valid_move_text(text))
        .ok_or_else(|| invalid("invalid teacher bestmove"))?;
    let mv = Move::from_uci(text).ok_or_else(|| invalid("invalid teacher move coordinates"))?;
    if !legal_moves.contains(&mv) {
        return Err(invalid(format!(
            "teacher bestmove {text} is outside legal searchmoves"
        )));
    }
    Ok(mv)
}

fn validate_position_command(command: &str) -> io::Result<()> {
    if command.contains(['\r', '\n', '\0']) {
        return Err(invalid("invalid multiline teacher position"));
    }
    let words = command.split_whitespace().collect::<Vec<_>>();
    if words.first() != Some(&"position") {
        return Err(invalid("teacher requires a position command"));
    }
    let moves_at = words
        .iter()
        .position(|word| *word == "moves")
        .unwrap_or(words.len());
    match words.get(1).copied() {
        Some("startpos") if moves_at == 2 => {}
        Some("fen") if moves_at >= 4 => {
            Position::from_fen(&words[2..moves_at].join(" ")).map_err(invalid)?;
        }
        _ => return Err(invalid("invalid teacher position syntax")),
    }
    if moves_at < words.len()
        && !words[moves_at + 1..]
            .iter()
            .all(|text| valid_move_text(text))
    {
        return Err(invalid("invalid teacher history move"));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parser_enforces_whitelist_and_rejects_nonmoves() {
        let allowed = [Move::from_uci("a3a4").unwrap()];
        assert_eq!(
            parse_best_move("bestmove a3a4 ponder a6a5", &allowed).unwrap(),
            allowed[0]
        );
        for line in [
            "bestmove a3a5",
            "bestmove 0000",
            "bestmove (none)",
            "bestmove 帅五平四",
            "bestmove",
            "info bestmove a3a4",
        ] {
            assert!(parse_best_move(line, &allowed).is_err(), "{line}");
        }
    }

    #[test]
    fn strict_handshake_and_history_command() {
        let mut input = Vec::new();
        let mut output =
            Cursor::new(b"id name fake\noption name Hash type spin default 16\nuciok\n");
        let reply = transact(&mut input, &mut output, &["uci".into()], ResponseKind::Uci).unwrap();
        assert_eq!(input, b"uci\n");
        assert!(reply.options.contains("Hash"));
        assert!(validate_position_command("position startpos moves a3a4 a6a5").is_ok());
        assert!(validate_position_command("position startpos").is_ok());
        assert!(validate_position_command("position startpos\ngo infinite").is_err());
        assert!(validate_position_command("position startpos garbage").is_err());
        assert!(
            transact(
                &mut Vec::new(),
                &mut Cursor::new(b"not-uciok\n"),
                &["uci".into()],
                ResponseKind::Uci
            )
            .is_err()
        );
    }

    #[test]
    fn fake_engine_process() {
        if std::env::var_os("CHINESEAI_FAKE_TEACHER_TEST").is_none() {
            return;
        }
        for line in io::stdin().lock().lines() {
            let line = line.unwrap();
            if line == "uci" {
                println!(
                    "option name Threads type spin default 1\noption name Hash type spin default 16\noption name EvalFile type string default none\nuciok"
                );
            } else if line == "isready" {
                println!("readyok");
            }
            io::stdout().flush().unwrap();
        }
    }

    #[test]
    fn timeout_kills_process_and_disables_reuse() {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "pikafish::teacher::tests::fake_engine_process",
                "--nocapture",
            ])
            .env("CHINESEAI_FAKE_TEACHER_TEST", "1");
        let mut teacher =
            PikafishTeacher::launch(command, "unused.nnue", 16, Duration::from_secs(10)).unwrap();
        teacher.timeout = Duration::from_millis(30);
        let legal = Position::startpos().legal_moves();
        let error = teacher
            .best_move("position startpos", &legal, TeacherBudget::Depth(1))
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(teacher.child.try_wait().unwrap().is_some());
        assert_eq!(
            teacher.new_game().unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
    }

    #[test]
    #[ignore = "requires local tools/pikafish.exe and pikafish.nnue"]
    fn local_pikafish_smoke() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config = TeacherConfig {
            exe: root.join("tools/pikafish.exe"),
            nnue: root.join("tools/pikafish.nnue"),
            hash_mb: 16,
            timeout: Duration::from_secs(30),
        };
        let mut teacher = PikafishTeacher::spawn(&config).unwrap();
        let pid = teacher.child.id();
        teacher.new_game().unwrap();
        let legal = Position::startpos().legal_moves();
        let mv = teacher
            .best_move("position startpos", &legal, TeacherBudget::Depth(2))
            .unwrap();
        assert!(legal.contains(&mv));
        assert_eq!(
            teacher
                .best_move("position startpos", &[legal[0]], TeacherBudget::Nodes(100))
                .unwrap(),
            legal[0]
        );
        assert_eq!(teacher.child.id(), pid);
    }
}
