//! 整盘固定的杀法指令。资源不足时关闭偏好，不移除任何合法着法。
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::xiangqi::{
    Color, MateClassification, MateKind, MatingPattern, Move, PieceKind, Position,
};

macro_rules! mating_targets {
    ($($variant:ident => $key:literal),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        #[repr(u8)]
        pub enum MatingTarget { #[default] None, $($variant),+ }

        impl MatingTarget {
            pub const ALL: [Self; 14] = [$(Self::$variant),+];
            pub fn index(self) -> Option<usize> {
                (self != Self::None).then_some(self as usize).map(|code| code - 1)
            }
            pub fn pattern(self) -> Option<MatingPattern> {
                match self { Self::None => None, $(Self::$variant => Some(MatingPattern::$variant)),+ }
            }
            pub fn name(self) -> &'static str {
                self.pattern().map_or("普通争胜", MatingPattern::name)
            }
            pub fn key(self) -> &'static str {
                match self { Self::None => "none", $(Self::$variant => $key),+ }
            }
            pub fn from_code(code: u8) -> Option<Self> {
                if code == 0 { Some(Self::None) } else { Self::ALL.get(code as usize - 1).copied() }
            }
        }
    };
}

// 前两项及回放编号保留，后续目标只在末尾追加。
mating_targets! {
    DoubleCannon => "double_cannon", IronBolt => "iron_bolt",
    HorseCannon => "horse_cannon", HeavenEarthCannon => "heaven_earth_cannon",
    SmotheredCannon => "smothered_cannon",
    WhiteFacedGeneral => "white_faced_general", ElbowHorse => "elbow_horse",
    PalaceCornerHorse => "palace_corner_horse", AnglerHorse => "angler_horse",
    OctagonalHorse => "octagonal_horse", HighAnglerHorse => "high_angler_horse",
    DoubleHorse => "double_horse", CrossRooks => "cross_rooks", TwoGhosts => "two_ghosts",
}

pub const MATING_TARGET_COUNT: usize = MatingTarget::ALL.len();
pub const GOAL_CONTEXT_SIZE: usize = 2 * MATING_TARGET_COUNT;

impl FromStr for MatingTarget {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" | "普通" => return Ok(Self::None),
            "铁门闩" => return Ok(Self::IronBolt),
            "侧面虎" => return Ok(Self::HighAnglerHorse),
            _ => {}
        }
        let key = value.replace('-', "_");
        Self::ALL
            .into_iter()
            .find(|target| target.key() == key || target.name() == value)
            .ok_or_else(|| {
                format!("不支持的杀法目标 `{value}`，可选 none 或 MatingTarget 列出的杀法")
            })
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
        let mut horses = 0;
        let mut rooks = 0;
        let mut attackers = 0;
        for sq in position.occupied_squares() {
            let piece = position.piece_at(sq).unwrap();
            if piece.color == self.side {
                cannons += usize::from(piece.kind == PieceKind::Cannon);
                horses += usize::from(piece.kind == PieceKind::Horse);
                rooks += usize::from(piece.kind == PieceKind::Rook);
                attackers += usize::from(piece.kind != PieceKind::General);
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
            MatingTarget::HorseCannon => cannons >= 1 && horses >= 1,
            MatingTarget::HeavenEarthCannon => cannons >= 2,
            MatingTarget::SmotheredCannon => cannons >= 1,
            MatingTarget::TwoGhosts => finishers >= 2,
            MatingTarget::DoubleHorse => horses >= 2,
            MatingTarget::CrossRooks => rooks >= 2,
            MatingTarget::ElbowHorse
            | MatingTarget::PalaceCornerHorse
            | MatingTarget::AnglerHorse
            | MatingTarget::OctagonalHorse
            | MatingTarget::HighAnglerHorse => horses >= 1,
            MatingTarget::WhiteFacedGeneral => attackers >= 1,
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

    /// [己方各目标、对方各目标]，顺序按 MatingTarget::ALL，包含偏好强度。
    pub fn features(self, position: &Position) -> [f32; GOAL_CONTEXT_SIZE] {
        let goal = self.effective(position);
        let mut values = [0.0; GOAL_CONTEXT_SIZE];
        if goal.enabled() {
            let role = usize::from(goal.side != position.side_to_move());
            values[role * MATING_TARGET_COUNT + goal.target.index().unwrap()] = goal.strength;
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
        let mut cannons = [0; 2];
        let mut cannon_count = 0;
        let mut approach = 0.0f32;
        let mut horse = 0.0f32;
        let mut horse_sum = 0.0f32;
        let mut rook = 0.0f32;
        let mut soldier = 0.0f32;
        let mut near = 0.0f32;
        let extended = !matches!(
            self.target,
            MatingTarget::DoubleCannon | MatingTarget::IronBolt
        );
        let closeness = |sq: usize, file: usize, rank: usize| {
            1.0 - ((sq % 9).abs_diff(file) + (sq / 9).abs_diff(rank)) as f32 / 17.0
        };
        for sq in position.occupied_squares() {
            let piece = position.piece_at(sq).unwrap();
            if piece.color != self.side {
                continue;
            }
            if piece.kind == PieceKind::Cannon {
                if cannon_count < 2 {
                    cannons[cannon_count] = sq;
                    cannon_count += 1;
                }
            }
            if extended {
                let proximity = closeness(sq, king_file, king / 9);
                if piece.kind != PieceKind::General {
                    near = near.max(proximity);
                }
                match piece.kind {
                    PieceKind::Horse => {
                        let (files, rank) = match self.target {
                            MatingTarget::ElbowHorse => ([2, 6], 1),
                            MatingTarget::OctagonalHorse
                                if matches!(king_file, 3 | 5)
                                    && matches!((king / 9).abs_diff(home), 0 | 2) =>
                            {
                                ([8 - king_file; 2], 2 - (king / 9).abs_diff(home))
                            }
                            MatingTarget::PalaceCornerHorse | MatingTarget::OctagonalHorse => {
                                ([3, 5], 2)
                            }
                            MatingTarget::AnglerHorse => ([2, 6], 2),
                            MatingTarget::HighAnglerHorse => ([2, 6], 3),
                            MatingTarget::HorseCannon => ([king_file; 2], 2),
                            _ => ([king_file; 2], 1),
                        };
                        let rank = if self.side == Color::Red {
                            rank
                        } else {
                            9 - rank
                        };
                        let score = files
                            .into_iter()
                            .map(|f| closeness(sq, f, rank))
                            .fold(0.0, f32::max);
                        horse = horse.max(score);
                        horse_sum += score;
                    }
                    PieceKind::Rook => {
                        rook += proximity;
                        soldier += [3, 5]
                            .into_iter()
                            .map(|f| closeness(sq, f, home))
                            .fold(0.0, f32::max);
                    }
                    PieceKind::Soldier => {
                        soldier += [3, 5]
                            .into_iter()
                            .map(|f| closeness(sq, f, home))
                            .fold(0.0, f32::max);
                    }
                    _ => {}
                }
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
                let [a, b] = cannons;
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
                let cannon = cannons[..cannon_count]
                    .iter()
                    .copied()
                    .map(|sq| {
                        0.4 * (1.0 - (sq % 9).abs_diff(4) as f32 / 4.0)
                            + 0.1 * (1.0 - (sq / 9).abs_diff(king / 9) as f32 / 9.0)
                    })
                    .fold(0.0f32, f32::max);
                cannon + 0.4 * approach
            }
            MatingTarget::HorseCannon => {
                let cannon = cannons[..cannon_count]
                    .iter()
                    .map(|&sq| closeness(sq, king_file, king / 9))
                    .fold(0.0, f32::max);
                0.5 * horse + 0.3 * cannon
            }
            MatingTarget::HeavenEarthCannon => {
                let central = cannons
                    .iter()
                    .map(|&sq| closeness(sq, 4, king / 9))
                    .fold(0.0, f32::max);
                let bottom = cannons
                    .iter()
                    .map(|&sq| closeness(sq, king_file, home))
                    .fold(0.0, f32::max);
                0.4 * central + 0.4 * bottom
            }
            MatingTarget::SmotheredCannon => {
                0.8 * cannons[..cannon_count]
                    .iter()
                    .map(|&sq| closeness(sq, king_file, home))
                    .fold(0.0, f32::max)
            }
            MatingTarget::ElbowHorse
            | MatingTarget::PalaceCornerHorse
            | MatingTarget::AnglerHorse
            | MatingTarget::OctagonalHorse
            | MatingTarget::HighAnglerHorse => 0.8 * horse,
            MatingTarget::DoubleHorse => 0.4 * horse_sum,
            MatingTarget::CrossRooks => 0.4 * rook,
            MatingTarget::TwoGhosts => 0.4 * soldier,
            MatingTarget::WhiteFacedGeneral => {
                let general = position.general_square(self.side).unwrap();
                0.4 * (1.0 - (general % 9).abs_diff(king_file) as f32 / 8.0) + 0.4 * near
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_target_has_unique_role_encoding_and_symmetric_finite_opening_prior() {
        let start = Position::startpos();
        let mv = start.parse_uci_move("b2e2").unwrap();
        let mirror = start.mirror_files();
        let mirrored_move = mirror.parse_uci_move("h2e2").unwrap();
        for (index, target) in MatingTarget::ALL.into_iter().enumerate() {
            assert_eq!(target.index(), Some(index));
            assert_eq!(target.pattern(), Some(MatingPattern::ALL[index]));
            assert_eq!(MatingTarget::from_code(target as u8), Some(target));
            for name in [
                target.key().to_string(),
                target.key().replace('_', "-"),
                target.name().to_string(),
            ] {
                assert_eq!(name.parse::<MatingTarget>().unwrap(), target);
            }
            for side in [Color::Red, Color::Black] {
                let goal = AzMatingGoal {
                    target,
                    side,
                    strength: 0.8,
                };
                let features = goal.features(&start);
                assert_eq!(features.iter().filter(|&&v| v != 0.0).count(), 1);
                assert_eq!(
                    features[index + usize::from(side == Color::Black) * MATING_TARGET_COUNT],
                    0.8
                );
                let bias = goal.policy_bias(&start, mv);
                assert!(bias.is_finite() && (-0.5..=0.5).contains(&bias));
                assert!((bias - goal.policy_bias(&mirror, mirrored_move)).abs() < 1e-6);
            }
        }
        assert_eq!(MatingTarget::from_code(255), None);
        for removed in [
            "double_check",
            "rook_check",
            "horse_check",
            "cannon_check",
            "soldier_check",
            "smothered_mate",
            "double_rook",
            "three_pieces_to_side",
        ] {
            assert!(removed.parse::<MatingTarget>().is_err());
        }
        assert_eq!(
            "侧面虎".parse::<MatingTarget>().unwrap(),
            MatingTarget::HighAnglerHorse
        );
    }

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
        let mut expected = [0.0; GOAL_CONTEXT_SIZE];
        expected[1] = 0.6;
        assert_eq!(goal.features(&Position::startpos()), expected);
        let black =
            Position::from_fen(&Position::startpos().to_fen().replace(" w ", " b ")).unwrap();
        expected[1] = 0.0;
        expected[MATING_TARGET_COUNT + 1] = 0.6;
        assert_eq!(goal.features(&black), expected);
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
