use std::{
    io::{self, BufRead, BufReader, Write},
    process::{Child, Command, Stdio},
};
pub struct Engine {
    child: Child,
    output: BufReader<std::process::ChildStdout>,
}
#[derive(Clone, Debug)]
pub struct Score {
    pub cp: i32,
    pub mate: Option<i32>,
    pub best: String,
    pub pv: Vec<String>,
}
impl Engine {
    pub fn new(path: &str) -> io::Result<Self> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let output = BufReader::new(child.stdout.take().unwrap());
        let mut e = Self { child, output };
        e.send("uci")?;
        e.wait("uciok")?;
        e.send("setoption name Threads value 1")?;
        e.send("isready")?;
        e.wait("readyok")?;
        Ok(e)
    }
    fn send(&mut self, line: &str) -> io::Result<()> {
        let input = self.child.stdin.as_mut().unwrap();
        writeln!(input, "{line}")?;
        input.flush()
    }
    fn line(&mut self) -> io::Result<String> {
        let mut line = String::new();
        if self.output.read_line(&mut line)? == 0 {
            return Err(io::Error::other("Pikafish EOF"));
        }
        Ok(line)
    }
    fn wait(&mut self, token: &str) -> io::Result<()> {
        while self.line()?.trim() != token {}
        Ok(())
    }
    pub fn score(
        &mut self,
        command: &str,
        depth: usize,
        forced: Option<&str>,
    ) -> io::Result<Score> {
        self.send("setoption name Clear Hash")?;
        self.send(command)?;
        self.send(&format!(
            "go depth {depth}{}",
            forced
                .map(|m| format!(" searchmoves {m}"))
                .unwrap_or_default()
        ))?;
        let mut out = Score {
            cp: 0,
            mate: None,
            best: String::new(),
            pv: vec![],
        };
        loop {
            let line = self.line()?;
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            if tokens.first() == Some(&"bestmove") {
                out.best = tokens.get(1).unwrap_or(&"").to_string();
                return Ok(out);
            }
            if tokens.first() != Some(&"info") || !tokens.contains(&"pv") {
                continue;
            }
            if let Some(i) = tokens.iter().position(|&s| s == "pv") {
                out.pv = tokens[i + 1..].iter().map(|s| s.to_string()).collect();
            }
            if let Some(i) = tokens.iter().position(|&s| s == "score") {
                if let Some(v) = tokens.get(i + 2).and_then(|v| v.parse::<i32>().ok()) {
                    if tokens.get(i + 1) == Some(&"mate") {
                        out.cp = v.signum() * 30000;
                        out.mate = Some(v);
                    } else if tokens.get(i + 1) == Some(&"cp") {
                        out.cp = v;
                        out.mate = None;
                    }
                }
            }
        }
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}
