use crate::nnue::HistoryMove;
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

use super::alphazero::append_history;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AzRuleSet {
    Simple,
    Full,
}

#[derive(Clone, Debug)]
pub struct AzEnv {
    position: Position,
    history: Vec<HistoryMove>,
    rule_history: Vec<RuleHistoryEntry>,
    rules: AzRuleSet,
}

impl AzEnv {
    pub fn startpos(rules: AzRuleSet) -> Self {
        Self::from_position(Position::startpos(), rules)
    }

    pub fn from_position(position: Position, rules: AzRuleSet) -> Self {
        let rule_history = position.initial_rule_history();
        Self {
            position,
            history: Vec::new(),
            rule_history,
            rules,
        }
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn history(&self) -> &[HistoryMove] {
        &self.history
    }

    pub fn rule_history(&self) -> &[RuleHistoryEntry] {
        &self.rule_history
    }

    pub fn rule_history_vec(&self) -> Vec<RuleHistoryEntry> {
        self.rule_history.clone()
    }

    pub fn rules(&self) -> AzRuleSet {
        self.rules
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        self.position.legal_moves()
    }

    pub fn terminal_value(&self) -> Option<f32> {
        terminal_value_for(&self.position, &self.rule_history, self.rules)
    }

    pub fn game_result(&self) -> Option<f32> {
        game_result_for(&self.position, &self.rule_history, self.rules)
    }

    pub fn make_move(&mut self, mv: Move) {
        append_history(&mut self.history, &self.position, mv);
        self.rule_history
            .push(self.position.rule_history_entry_after_move(mv));
        self.position.make_move(mv);
    }
}

pub(super) fn terminal_value_for(
    position: &Position,
    rule_history: &[RuleHistoryEntry],
    rules: AzRuleSet,
) -> Option<f32> {
    if !position.has_general(Color::Red) {
        return Some(if position.side_to_move() == Color::Red {
            -1.0
        } else {
            1.0
        });
    }
    if !position.has_general(Color::Black) {
        return Some(if position.side_to_move() == Color::Black {
            -1.0
        } else {
            1.0
        });
    }
    if !position.has_dynamic_material(Color::Red) && !position.has_dynamic_material(Color::Black) {
        return Some(0.0);
    }
    if rules == AzRuleSet::Full {
        if let Some(outcome) = position.rule_outcome_with_history(rule_history) {
            return Some(match outcome {
                RuleOutcome::Draw(_) => 0.0,
                RuleOutcome::Win(color) => {
                    if color == position.side_to_move() {
                        1.0
                    } else {
                        -1.0
                    }
                }
            });
        }
    }
    None
}

pub(super) fn game_result_for(
    position: &Position,
    rule_history: &[RuleHistoryEntry],
    rules: AzRuleSet,
) -> Option<f32> {
    if !position.has_general(Color::Red) {
        return Some(-1.0);
    }
    if !position.has_general(Color::Black) {
        return Some(1.0);
    }
    if position.legal_moves().is_empty() {
        return Some(if position.side_to_move() == Color::Red {
            -1.0
        } else {
            1.0
        });
    }
    if !position.has_dynamic_material(Color::Red) && !position.has_dynamic_material(Color::Black) {
        return Some(0.0);
    }
    if rules == AzRuleSet::Full {
        if let Some(outcome) = position.rule_outcome_with_history(rule_history) {
            return Some(match outcome {
                RuleOutcome::Draw(_) => 0.0,
                RuleOutcome::Win(Color::Red) => 1.0,
                RuleOutcome::Win(Color::Black) => -1.0,
            });
        }
    }
    None
}
