//! 整盘固定的杀法指令。资源不足时关闭偏好，不移除任何合法着法。
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::xiangqi::{
    Color, MateClassification, MateKind, MatingPattern, Move, PieceKind, Position,
};

/// 仅用于成功轨迹的策略模仿；不改目标概率、MCTS 价值或 WDL 标签。
/// 100 半回合约为 0.74，200 半回合约为 0.55，不把长轨迹一刀切丢弃。
pub(crate) fn goal_imitation_discount(plies: f32) -> f32 {
    0.997_f32.powf(plies)
}

fn horse_finish_distances() -> &'static [[u8; 90]; MATING_TARGET_COUNT * 2] {
    static TABLE: std::sync::OnceLock<[[u8; 90]; MATING_TARGET_COUNT * 2]> =
        std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0; 90]; MATING_TARGET_COUNT * 2];
        for color in [Color::Red, Color::Black] {
            for target in MatingTarget::ALL {
                let squares: &[(usize, usize)] = match target {
                    MatingTarget::HighAnglerHorse => &[(2, 3), (6, 3)],
                    _ => continue,
                };
                let row = usize::from(color == Color::Black) * MATING_TARGET_COUNT
                    + target.index().unwrap();
                table[row].fill(u8::MAX);
                let mut queue = std::collections::VecDeque::new();
                for &(file, rank) in squares {
                    let sq = (if color == Color::Red { rank } else { 9 - rank }) * 9 + file;
                    table[row][sq] = 0;
                    queue.push_back(sq);
                }
                while let Some(sq) = queue.pop_front() {
                    for (_, (dx, dy)) in crate::xiangqi::HORSE_STEPS {
                        let (x, y) = ((sq % 9) as i32 + dx, (sq / 9) as i32 + dy);
                        if (0..9).contains(&x) && (0..10).contains(&y) {
                            let next = (y * 9 + x) as usize;
                            if table[row][next] == u8::MAX {
                                table[row][next] = table[row][sq] + 1;
                                queue.push_back(next);
                            }
                        }
                    }
                }
            }
        }
        table
    })
}

macro_rules! mating_targets {
    ($($variant:ident => $key:literal),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        #[repr(u8)]
        pub enum MatingTarget { #[default] None, $($variant),+ }

        impl MatingTarget {
            pub const ALL: [Self; 4] = [$(Self::$variant),+];
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

// 当前 v50 回放的目标顺序；删除类别后已升级格式，禁止旧编号混用。
mating_targets! {
    DoubleCannon => "double_cannon", IronBolt => "iron_bolt",
    HorseCannon => "horse_cannon",
    HighAnglerHorse => "high_angler_horse",
}

pub const MATING_TARGET_COUNT: usize = MatingTarget::ALL.len();
pub const GOAL_CONTEXT_SIZE: usize = 2 * MATING_TARGET_COUNT;

impl FromStr for MatingTarget {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" | "普通" | "普通争胜" => return Ok(Self::None),
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

/// 同一局面的所有候选共享基准分和工作棋盘；每次试走后完整撤销。
pub(crate) struct MatingPolicyPrior {
    goal: AzMatingGoal,
    position: Position,
    before: f32,
}

impl MatingPolicyPrior {
    pub(crate) fn bias(&mut self, mv: Move) -> f32 {
        let undo = self.position.make_move(mv);
        let after = self.goal.progress(&self.position);
        self.position.unmake_move(mv, undo);
        (4.0 * self.goal.strength * (after - self.before)).clamp(-0.5, 0.5)
    }
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

    /// 非终局的乐观完成步数下界；忽略蹩腿和占位，只用于排除不可能更短的证明。
    pub(crate) fn minimum_finish_plies(self, position: &Position) -> u32 {
        let distance = if self.target == MatingTarget::HighAnglerHorse {
            let row = usize::from(self.side == Color::Black) * MATING_TARGET_COUNT
                + self.target.index().unwrap();
            position
                .occupied_squares()
                .filter(|&sq| {
                    position
                        .piece_at(sq)
                        .is_some_and(|p| p.color == self.side && p.kind == PieceKind::Horse)
                })
                .map(|sq| horse_finish_distances()[row][sq])
                .min()
                .unwrap_or(u8::MAX)
        } else {
            0
        };
        if distance == u8::MAX {
            return u32::MAX;
        }
        2 * u32::from(distance.max(1)) - u32::from(position.side_to_move() == self.side)
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
        let mut attackers = 0;
        for sq in position.occupied_squares() {
            let piece = position.piece_at(sq).unwrap();
            if piece.color == self.side {
                cannons += usize::from(piece.kind == PieceKind::Cannon);
                horses += usize::from(piece.kind == PieceKind::Horse);
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
            MatingTarget::HighAnglerHorse => horses >= 1 && attackers >= 2,
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
        self.policy_prior(position)
            .map_or(0.0, |mut prior| prior.bias(mv))
    }

    pub(crate) fn policy_prior(self, position: &Position) -> Option<MatingPolicyPrior> {
        let goal = self.effective(position);
        if !goal.enabled() || position.side_to_move() != goal.side {
            return None;
        }
        Some(MatingPolicyPrior {
            goal,
            position: position.clone(),
            before: goal.progress(position),
        })
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
            if extended && piece.kind == PieceKind::Horse {
                let score = if self.target == MatingTarget::HighAnglerHorse {
                    let row = usize::from(self.side == Color::Black) * MATING_TARGET_COUNT
                        + self.target.index().unwrap();
                    1.0 / (1.0 + f32::from(horse_finish_distances()[row][sq]))
                } else {
                    let rank = if self.side == Color::Red { 2 } else { 7 };
                    closeness(sq, king_file, rank)
                };
                horse = horse.max(score);
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
            MatingTarget::HighAnglerHorse => 0.8 * horse,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horse_prior_uses_real_knight_steps_instead_of_manhattan_distance() {
        let position = Position::from_fen("4k4/5R3/3a5/9/7N1/9/9/3K5/9/3A1A3 w").unwrap();
        let goal = AzMatingGoal {
            target: MatingTarget::HighAnglerHorse,
            side: Color::Red,
            strength: 0.4,
        };
        let closer = position.parse_uci_move("h5f4").unwrap();
        let misleading = position.parse_uci_move("h5f6").unwrap();
        assert!(position.legal_moves().contains(&closer));
        assert!(position.legal_moves().contains(&misleading));
        assert_eq!(goal.minimum_finish_plies(&position), 3);
        for (mv, expected) in [(closer, 2), (misleading, 6)] {
            let mut next = position.clone();
            next.make_move(mv);
            assert_eq!(goal.minimum_finish_plies(&next), expected);
        }
        assert!(goal.policy_bias(&position, closer) > 0.0);
        assert!(goal.policy_bias(&position, misleading) < 0.0);
    }

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
            "heaven_earth_cannon",
            "smothered_cannon",
            "天地炮",
            "闷宫",
            "two_ghosts",
            "two-ghosts",
            "二鬼拍门",
            "white_faced_general",
            "白脸将",
            "elbow_horse",
            "卧槽马",
            "palace_corner_horse",
            "挂角马",
            "angler_horse",
            "钓鱼马",
            "octagonal_horse",
            "八角马",
            "double_horse",
            "双马饮泉",
            "cross_rooks",
            "双车错",
        ] {
            assert!(removed.parse::<MatingTarget>().is_err());
        }
        assert_eq!(
            "侧面虎".parse::<MatingTarget>().unwrap(),
            MatingTarget::HighAnglerHorse
        );
    }

    #[test]
    fn prepared_prior_matches_full_move_reference_and_restores_all_board_state() {
        let mut positions = vec![Position::startpos()];
        let mut middle = Position::startpos();
        for text in ["h2e2", "h9g7", "b0c2", "b7b4", "a0b0", "b4e4"] {
            let mv = middle.parse_uci_move(text).unwrap();
            middle.make_move(mv);
        }
        positions.extend([
            middle,
            Position::from_fen("4ka3/4a4/9/9/9/6C2/9/9/9/5K3 w").unwrap(),
            Position::from_fen("4k4/4R4/9/9/9/9/9/9/9/3K5 w").unwrap(),
        ]);
        for position in positions {
            let moves = position.legal_moves();
            for target in MatingTarget::ALL {
                for strength in [0.0, 0.4, 1.0] {
                    for side in [Color::Red, Color::Black] {
                        let goal = AzMatingGoal {
                            target,
                            side,
                            strength,
                        };
                        let effective = goal.effective(&position);
                        let mut prepared = goal.policy_prior(&position);
                        for &mv in moves.iter().chain(moves.iter().rev()) {
                            let expected = if effective.enabled() && side == position.side_to_move()
                            {
                                let before = effective.progress(&position);
                                let mut after = position.clone();
                                after.make_move(mv);
                                (4.0 * strength * (effective.progress(&after) - before))
                                    .clamp(-0.5, 0.5)
                            } else {
                                0.0
                            };
                            let actual = prepared.as_mut().map_or(0.0, |prior| prior.bias(mv));
                            assert_eq!(actual, expected, "{target:?} {side:?} {mv}");
                            if let Some(prior) = &prepared {
                                assert_eq!(prior.position, position);
                            }
                        }
                    }
                }
            }
        }
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
