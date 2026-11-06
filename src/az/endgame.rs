//! 本地搜索教师的有界进程池；整条残局续弈独占实例。
//! 教师仅决定着法，胜负由调用方按本引擎规则走到终局确定。

use crate::{
    pikafish::teacher::{PikafishTeacher, TeacherBudget, TeacherConfig},
    xiangqi::{Move, Position, RuleHistoryEntry},
};
use std::{
    io,
    sync::{Condvar, Mutex},
};

#[derive(Debug)]
pub struct AzEndgameTeacherPool {
    state: Mutex<PoolState>,
    available: Condvar,
    nodes: u64,
    max_pieces: usize,
    max_plies: usize,
}

#[derive(Debug)]
struct PoolState {
    teachers: Vec<PikafishTeacher>,
    failure: Option<(io::ErrorKind, String)>,
}

pub struct AzEndgameTeacherLease<'a> {
    pool: &'a AzEndgameTeacherPool,
    teacher: Option<PikafishTeacher>,
}

impl AzEndgameTeacherPool {
    pub fn new(
        config: TeacherConfig,
        processes: usize,
        nodes: u64,
        max_pieces: usize,
        max_plies: usize,
    ) -> io::Result<Self> {
        if processes == 0 || nodes == 0 || !(2..=32).contains(&max_pieces) || max_plies == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid endgame teacher limits",
            ));
        }
        let mut teachers = Vec::with_capacity(processes);
        for _ in 0..processes {
            let mut teacher = PikafishTeacher::spawn(&config)?;
            teacher.new_game()?;
            // isready 不保证 NNUE 已被实际读取；浅搜使缺失/错误权重在训练前失败。
            teacher.best_move(
                "position startpos",
                &Position::startpos().legal_moves(),
                TeacherBudget::Nodes(1),
            )?;
            teachers.push(teacher);
        }
        Ok(Self {
            state: Mutex::new(PoolState {
                teachers,
                failure: None,
            }),
            available: Condvar::new(),
            nodes,
            max_pieces,
            max_plies,
        })
    }

    /// 子数包括双方将帅、士象和兵卒。
    pub fn should_take_over(&self, position: &Position) -> bool {
        position.occupied_squares().count() <= self.max_pieces
    }

    pub fn max_plies(&self) -> usize {
        self.max_plies
    }

    pub fn acquire(&self) -> io::Result<AzEndgameTeacherLease<'_>> {
        let mut state = self.state.lock().map_err(|_| poisoned())?;
        let mut teacher = loop {
            if let Some((kind, message)) = &state.failure {
                return Err(io::Error::new(*kind, message.clone()));
            }
            if let Some(teacher) = state.teachers.pop() {
                break teacher;
            }
            state = self.available.wait(state).map_err(|_| poisoned())?;
        };
        drop(state);
        if let Err(error) = teacher.new_game() {
            self.fail(&error);
            return Err(error);
        }
        self.check_failure()?;
        Ok(AzEndgameTeacherLease {
            pool: self,
            teacher: Some(teacher),
        })
    }

    fn fail(&self, error: &io::Error) {
        if let Ok(mut state) = self.state.lock() {
            state
                .failure
                .get_or_insert_with(|| (error.kind(), error.to_string()));
        }
        self.available.notify_all();
    }

    fn check_failure(&self) -> io::Result<()> {
        let state = self.state.lock().map_err(|_| poisoned())?;
        match &state.failure {
            Some((kind, message)) => Err(io::Error::new(*kind, message.clone())),
            None => Ok(()),
        }
    }
}

impl AzEndgameTeacherLease<'_> {
    pub fn best_move(&mut self, position_command: &str, legal: &[Move]) -> io::Result<Move> {
        self.pool.check_failure()?;
        let result = self
            .teacher
            .as_mut()
            .expect("teacher lease is active")
            .best_move(
                position_command,
                legal,
                TeacherBudget::Nodes(self.pool.nodes),
            );
        if let Err(error) = &result {
            self.pool.fail(error);
        }
        // 另一实例在本次搜索期间失败时，本实例也不得继续写入训练样本。
        self.pool.check_failure()?;
        result
    }
}

impl Drop for AzEndgameTeacherLease<'_> {
    fn drop(&mut self) {
        let Some(teacher) = self.teacher.take() else {
            return;
        };
        if let Ok(mut state) = self.pool.state.lock()
            && state.failure.is_none()
        {
            state.teachers.push(teacher);
        }
        self.pool.available.notify_one();
    }
}

/// 保留原始 FEN 和完整着法；不以仅含当前棋盘的 FEN 丢弃长将、长捉历史。
pub fn teacher_position_command(
    position: &Position,
    history: &[RuleHistoryEntry],
) -> io::Result<String> {
    let initial = position
        .initial_position_from_history(history)
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    let mut command = format!("position fen {}", initial.to_fen());
    if history.len() > 1 {
        command.push_str(" moves");
        for entry in &history[1..] {
            command.push(' ');
            command.push_str(&entry.mv.expect("history was validated").to_uci());
        }
    }
    Ok(command)
}

fn poisoned() -> io::Error {
    io::Error::other("endgame teacher pool poisoned")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Barrier, thread, time::Duration};

    fn empty_pool(max_pieces: usize) -> AzEndgameTeacherPool {
        AzEndgameTeacherPool {
            state: Mutex::new(PoolState {
                teachers: Vec::new(),
                failure: None,
            }),
            available: Condvar::new(),
            nodes: 100,
            max_pieces,
            max_plies: 120,
        }
    }

    fn play(position: &mut Position, history: &mut Vec<RuleHistoryEntry>, moves: &[&str]) {
        for text in moves {
            let mv = position.parse_uci_move(text).unwrap();
            assert!(position.legal_moves_with_rules(history).contains(&mv));
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
        }
    }

    #[test]
    fn command_preserves_full_repetition_history_and_raw_initial_clock() {
        let initial = Position::from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 17 1",
        )
        .unwrap();
        let mut position = initial.clone();
        let mut history = position.initial_rule_history();
        play(
            &mut position,
            &mut history,
            &["b0c2", "b9c7", "c2b0", "c7b9"],
        );
        assert_eq!(position.halfmove_clock(), 21);
        assert_eq!(
            position.initial_position_from_history(&history).unwrap(),
            initial
        );
        assert_eq!(
            teacher_position_command(&position, &history).unwrap(),
            format!(
                "position fen {} moves b0c2 b9c7 c2b0 c7b9",
                initial.to_fen()
            )
        );
    }

    #[test]
    fn command_restores_captured_piece_and_clock_before_capture() {
        let initial = Position::from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 17 1",
        )
        .unwrap();
        let mut position = initial.clone();
        let mut history = position.initial_rule_history();
        play(&mut position, &mut history, &["b2b9"]);
        assert_eq!(position.halfmove_clock(), 0);
        assert!(history[1].captured.is_some());
        assert_eq!(
            teacher_position_command(&position, &history).unwrap(),
            format!("position fen {} moves b2b9", initial.to_fen())
        );
        assert_eq!(
            position.initial_position_from_history(&history).unwrap(),
            initial
        );
    }

    #[test]
    fn invalid_history_is_rejected_instead_of_silently_resetting_context() {
        let mut position = Position::startpos();
        let mut history = position.initial_rule_history();
        play(&mut position, &mut history, &["b0c2", "b9c7"]);
        assert!(teacher_position_command(&position, &[]).is_err());
        for field in 0..5 {
            let mut bad = history.clone();
            match field {
                0 => bad[1].mv = None,
                1 => bad[1].hash ^= 1,
                2 => bad[2].rule60_clock += 1,
                3 => bad[2].mv = Some(Move::new(255, 255)),
                _ => bad[1].gives_check = !bad[1].gives_check,
            }
            assert!(
                teacher_position_command(&position, &bad).is_err(),
                "field {field}"
            );
        }
    }

    #[test]
    fn takeover_counts_all_pieces_including_generals() {
        let position = Position::from_fen("4k4/9/9/9/9/9/9/4R4/9/3K5 w").unwrap();
        assert!(!empty_pool(2).should_take_over(&position));
        assert!(empty_pool(3).should_take_over(&position));
        assert!(!empty_pool(3).should_take_over(&Position::startpos()));
    }

    #[test]
    fn one_fatal_error_wakes_all_waiters_and_keeps_first_cause() {
        let pool = empty_pool(14);
        let barrier = Barrier::new(3);
        thread::scope(|scope| {
            let waiters = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        pool.acquire().err().expect("failed pool must reject lease")
                    })
                })
                .collect::<Vec<_>>();
            barrier.wait();
            pool.fail(&io::Error::new(
                io::ErrorKind::TimedOut,
                "first teacher timeout",
            ));
            pool.fail(&io::Error::other("later failure"));
            for waiter in waiters {
                let error = waiter.join().unwrap();
                assert_eq!(error.kind(), io::ErrorKind::TimedOut);
                assert_eq!(error.to_string(), "first teacher timeout");
            }
        });
        assert!(pool.check_failure().is_err());
    }

    #[test]
    fn invalid_limits_fail_before_starting_processes() {
        let config = TeacherConfig {
            exe: "missing-teacher".into(),
            nnue: "missing-nnue".into(),
            hash_mb: 16,
            timeout: Duration::from_secs(1),
        };
        for (processes, nodes, pieces, plies) in [
            (0, 1, 14, 120),
            (1, 0, 14, 120),
            (1, 1, 33, 120),
            (1, 1, 14, 0),
        ] {
            let error = AzEndgameTeacherPool::new(config.clone(), processes, nodes, pieces, plies)
                .unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        }
    }
}
