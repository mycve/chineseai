use super::{Color, Move, Position, RuleHistoryEntry, RuleOutcome, Undo};

#[derive(Clone, Copy, Debug)]
pub struct AppliedMove {
    pub mv: Move,
    undo: Undo,
    rule_history_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepOutcome {
    Ongoing,
    Draw(super::RuleDrawReason),
    Win(Color),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IllegalMove {
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct XiangqiEnv {
    position: Position,
    rule_history: Vec<RuleHistoryEntry>,
    history: Vec<AppliedMove>,
}

impl Default for XiangqiEnv {
    fn default() -> Self {
        Self::startpos()
    }
}

impl XiangqiEnv {
    pub fn startpos() -> Self {
        Self::from_position(Position::startpos())
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        Ok(Self::from_position(Position::from_fen(fen)?))
    }

    pub fn from_position(position: Position) -> Self {
        let rule_history = position.initial_rule_history();
        Self {
            position,
            rule_history,
            history: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn position(&self) -> &Position {
        &self.position
    }

    #[inline(always)]
    pub fn rule_history(&self) -> &[RuleHistoryEntry] {
        &self.rule_history
    }

    #[inline(always)]
    pub fn history(&self) -> &[AppliedMove] {
        &self.history
    }

    pub fn reset(&mut self, position: Position) {
        *self = Self::from_position(position);
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        self.position.legal_moves_with_rules(&self.rule_history)
    }

    pub fn raw_legal_moves(&self) -> Vec<Move> {
        self.position.legal_moves()
    }

    pub fn parse_uci_move(&self, text: &str) -> Option<Move> {
        self.position.parse_uci_move(text)
    }

    pub fn step_uci(&mut self, text: &str) -> Result<StepOutcome, IllegalMove> {
        let mv = self.parse_uci_move(text).ok_or_else(|| IllegalMove {
            text: text.to_string(),
        })?;
        self.step(mv).map_err(|_| IllegalMove {
            text: text.to_string(),
        })
    }

    pub fn step(&mut self, mv: Move) -> Result<StepOutcome, Move> {
        if !self.legal_moves().contains(&mv) {
            return Err(mv);
        }

        let rule_history_len = self.rule_history.len();
        self.rule_history
            .push(self.position.rule_history_entry_after_move(mv));
        let undo = self.position.make_move(mv);
        self.history.push(AppliedMove {
            mv,
            undo,
            rule_history_len,
        });

        Ok(self.current_outcome())
    }

    pub fn undo(&mut self) -> Option<Move> {
        let applied = self.history.pop()?;
        self.position.unmake_move(applied.mv, applied.undo);
        self.rule_history.truncate(applied.rule_history_len);
        Some(applied.mv)
    }

    pub fn current_outcome(&self) -> StepOutcome {
        match self.position.rule_outcome_with_history(&self.rule_history) {
            Some(RuleOutcome::Draw(reason)) => StepOutcome::Draw(reason),
            Some(RuleOutcome::Win(color)) => StepOutcome::Win(color),
            None if self.legal_moves().is_empty() => {
                StepOutcome::Win(self.position.side_to_move().opposite())
            }
            None => StepOutcome::Ongoing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_tracks_rule_history_while_stepping_and_undoing() {
        let mut env = XiangqiEnv::from_fen(
            "2Rakab2/8r/4c1n2/p3p1p1p/2p6/9/P3P3P/1CN1NC3/9/1RBAKArc1 b - - 0 1",
        )
        .unwrap();
        let initial_history_len = env.rule_history().len();
        let mv = env.parse_uci_move("g0g1").unwrap();

        assert_eq!(env.step(mv), Ok(StepOutcome::Ongoing));
        assert_eq!(env.history().len(), 1);
        assert_eq!(env.rule_history().len(), initial_history_len + 1);

        assert_eq!(env.undo(), Some(mv));
        assert!(env.history().is_empty());
        assert_eq!(env.rule_history().len(), initial_history_len);
    }
}
