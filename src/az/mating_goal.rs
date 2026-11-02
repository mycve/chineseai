//! 整盘固定的杀法指令。资源不足时关闭偏好，不移除任何合法着法。
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::xiangqi::{
    Color, MateClassification, MateKind, MatingPattern, Move, PieceKind, Position,
};

pub const GOAL_CONTEXT_SIZE: usize = 4;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatingTarget {
    #[default]
    None,
    DoubleCannon,
    IronBolt,
}

impl MatingTarget {
    pub fn index(self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::DoubleCannon => Some(0),
            Self::IronBolt => Some(1),
        }
    }

    pub fn pattern(self) -> Option<MatingPattern> {
        match self {
            Self::None => None,
            Self::DoubleCannon => Some(MatingPattern::DoubleCannon),
            Self::IronBolt => Some(MatingPattern::IronBolt),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "普通争胜",
            Self::DoubleCannon => "重炮",
            Self::IronBolt => "铁门栓",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DoubleCannon => "double_cannon",
            Self::IronBolt => "iron_bolt",
        }
    }
}

impl FromStr for MatingTarget {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" | "普通" => Ok(Self::None),
            "double_cannon" | "double-cannon" | "重炮" => Ok(Self::DoubleCannon),
            "iron_bolt" | "iron-bolt" | "铁门栓" | "铁门闩" => Ok(Self::IronBolt),
            _ => Err(format!(
                "不支持的杀法目标 `{value}`，请选择 none、double-cannon 或 iron-bolt"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AzMatingGoal {
    pub target: MatingTarget,
    /// 执行指令的一方，整盘固定；不能随搜索根节点换边。
    pub side: Color,
    pub strength: f32,
}

impl Default for AzMatingGoal {
    fn default() -> Self {
        Self::NONE
    }
}

impl AzMatingGoal {
    pub const NONE: Self = Self {
        target: MatingTarget::None,
        side: Color::Red,
        strength: 0.0,
    };

    pub fn validate(self) -> Result<(), String> {
        if !self.strength.is_finite() || !(0.0..=1.0).contains(&self.strength) {
            return Err("杀法偏好强度必须是 0..1 的有限数值".into());
        }
        Ok(())
    }

    pub fn enabled(self) -> bool {
        self.target != MatingTarget::None && self.strength > 0.0 && self.validate().is_ok()
    }

    /// 只用必要条件认定不可达，不能把搜索暂时没找到杀法视为不可能。
    pub fn effective(self, position: &Position) -> Self {
        if !self.enabled()
            || !position.has_general(self.side)
            || !position.has_general(self.side.opposite())
        {
            return Self::NONE;
        }
        let mut cannons = 0;
        let mut finishers = 0;
        let mut defenders = 0;
        for sq in position.occupied_squares() {
            let piece = position.piece_at(sq).unwrap();
            if piece.color == self.side {
                cannons += usize::from(piece.kind == PieceKind::Cannon);
                finishers +=
                    usize::from(matches!(piece.kind, PieceKind::Rook | PieceKind::Soldier));
            } else {
                defenders += usize::from(piece.kind != PieceKind::General);
            }
        }
        let possible = match self.target {
            MatingTarget::None => false,
            MatingTarget::DoubleCannon => cannons >= 2,
            MatingTarget::IronBolt => cannons >= 1 && finishers >= 1 && defenders >= 1,
        };
        if possible { self } else { Self::NONE }
    }

    /// 棋盘为当前行棋方 canonical 视角时，保持执行方身份对应关系。
    pub fn canonical_for(self, original_side: Color) -> Self {
        Self {
            side: if self.side == original_side {
                Color::Red
            } else {
                Color::Black
            },
            ..self
        }
    }

    /// [己方重炮、己方铁门栓、对方重炮、对方铁门栓]，数值包含偏好强度。
    pub fn features(self, position: &Position) -> [f32; GOAL_CONTEXT_SIZE] {
        let goal = self.effective(position);
        let mut values = [0.0; GOAL_CONTEXT_SIZE];
        if goal.enabled() {
            let role = usize::from(goal.side != position.side_to_move());
            values[role * 2 + goal.target.index().unwrap()] = goal.strength;
        }
        values
    }

    pub fn matches_mate(self, mate: &MateClassification) -> bool {
        self.enabled()
            && mate.kind == MateKind::Checkmate
            && mate.winner == self.side
            && self
                .target
                .pattern()
                .is_some_and(|p| mate.patterns.contains(&p))
    }

    /// 训练初期的小幅几何先验；不表示成功概率，不写入胜负证明或价值标签。
    /// 走子合法性由调用方负责。对手回合和资源不足时不改变先验。
    pub fn policy_bias(self, position: &Position, mv: Move) -> f32 {
        let goal = self.effective(position);
        if !goal.enabled() || position.side_to_move() != goal.side {
            return 0.0;
        }
        let before = goal.progress(position);
        let mut next = position.clone();
        next.make_move(mv);
        (4.0 * goal.strength * (goal.progress(&next) - before)).clamp(-0.5, 0.5)
    }

    fn progress(self, position: &Position) -> f32 {
        if !self.effective(position).enabled() {
            return 0.0;
        }
        let king = position.general_square(self.side.opposite()).unwrap();
        let king_file = king % 9;
        let home = if self.side == Color::Red { 0 } else { 9 };
        let mut cannons = Vec::with_capacity(2);
        let mut approach = 0.0f32;
        for sq in position.occupied_squares() {
            let piece = position.piece_at(sq).unwrap();
            if piece.color != self.side {
                continue;
            }
            if piece.kind == PieceKind::Cannon {
                cannons.push(sq);
            }
            if matches!(piece.kind, PieceKind::Rook | PieceKind::Soldier) {
                let distance =
                    (sq % 9).abs_diff(3).min((sq % 9).abs_diff(5)) + (sq / 9).abs_diff(home);
                approach = approach.max(1.0 - distance as f32 / 14.0);
            }
        }
        match self.target {
            MatingTarget::None => 0.0,
            MatingTarget::DoubleCannon => {
                let [a, b] = [cannons[0], cannons[1]];
                let alignment = if a % 9 == b % 9 {
                    0.35
                } else if a / 9 == b / 9 {
                    0.12
                } else {
                    0.0
                };
                alignment
                    + [a, b]
                        .into_iter()
                        .map(|sq| {
                            0.15 * (1.0 - (sq % 9).abs_diff(king_file) as f32 / 8.0)
                                + 0.1 * (1.0 - (sq / 9).abs_diff(king / 9) as f32 / 9.0)
                        })
                        .sum::<f32>()
            }
            MatingTarget::IronBolt => {
                let cannon = cannons
                    .into_iter()
                    .map(|sq| {
                        0.4 * (1.0 - (sq % 9).abs_diff(4) as f32 / 4.0)
                            + 0.1 * (1.0 - (sq / 9).abs_diff(king / 9) as f32 / 9.0)
                    })
                    .fold(0.0f32, f32::max);
                cannon + 0.4 * approach
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instructions_validate_strength_and_serialize_without_losing_mix_strength() {
        for strength in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(
                AzMatingGoal {
                    strength,
                    ..AzMatingGoal::NONE
                }
                .validate()
                .is_err()
            );
        }
        let goal = AzMatingGoal {
            target: MatingTarget::None,
            side: Color::Black,
            strength: 0.7,
        };
        let text = toml::to_string(&goal).unwrap();
        assert_eq!(toml::from_str::<AzMatingGoal>(&text).unwrap(), goal);
        assert!(!goal.enabled());
        assert_eq!(
            goal.features(&Position::startpos()),
            [0.0; GOAL_CONTEXT_SIZE]
        );
        assert_eq!(
            "重炮".parse::<MatingTarget>().unwrap(),
            MatingTarget::DoubleCannon
        );
    }

    #[test]
    fn role_strength_and_canonical_identity_are_explicit() {
        let goal = AzMatingGoal {
            target: MatingTarget::IronBolt,
            side: Color::Red,
            strength: 0.6,
        };
        assert_eq!(goal.features(&Position::startpos()), [0.0, 0.6, 0.0, 0.0]);
        let black =
            Position::from_fen(&Position::startpos().to_fen().replace(" w ", " b ")).unwrap();
        assert_eq!(goal.features(&black), [0.0, 0.0, 0.0, 0.6]);
        assert_eq!(goal.canonical_for(Color::Black).side, Color::Black);
    }

    #[test]
    fn necessary_resources_disable_preferences_and_bootstrap_is_only_a_prior() {
        let goal = AzMatingGoal {
            target: MatingTarget::DoubleCannon,
            side: Color::Red,
            strength: 0.8,
        };
        let start = Position::startpos();
        let central = start.parse_uci_move("h2e2").unwrap();
        assert!(goal.policy_bias(&start, central) > 0.0);
        assert_eq!(
            AzMatingGoal {
                strength: 0.0,
                ..goal
            }
            .policy_bias(&start, central),
            0.0
        );
        let missing = Position::from_fen("4k4/9/9/9/9/9/9/1C7/9/3K5 w").unwrap();
        assert_eq!(goal.effective(&missing), AzMatingGoal::NONE);
        assert_eq!(goal.features(&missing), [0.0; GOAL_CONTEXT_SIZE]);
        let unsupported = AzMatingGoal {
            target: MatingTarget::IronBolt,
            ..goal
        };
        assert_eq!(unsupported.effective(&missing), AzMatingGoal::NONE);
        assert!(start.classify_mate().is_none());
    }
}
