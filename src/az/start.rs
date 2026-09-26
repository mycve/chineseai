use crate::xiangqi::{Position, RuleHistoryEntry};

#[derive(Clone, Debug)]
pub struct AzStartSnapshot {
    pub position: Position,
    pub rule_history: Vec<RuleHistoryEntry>,
    pub phase_ply: u16,
    pub generation: u32,
}
