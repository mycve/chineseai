//! 终局杀形的多标签识别。定义、来源和刻意限制见 docs/mating-patterns.md。
//! 只使用棋盘合法着法，不把长将判负、搜索证明或步数截断当成将死。

use super::{
    Color, Move, PieceKind, Position, file_of, index, inside_palace, line_between_squares, rank_of,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MateKind {
    Checkmate,
    Stalemate,
}

/// 标签可以重叠；基础将军子标签与具名杀法也可以同时出现。
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MatingPattern {
    DoubleCannon,
    HorseCannon,
    IronBolt,
    SmotheredCannon,
    HeavenEarthCannon,
    TwoSoldiers,
    WhiteFacedGeneral,
    ElbowHorse,
    PalaceCornerHorse,
    AnglerHorse,
    OctagonalHorse,
    DoubleRook,
    SmotheredMate,
    DoubleCheck,
    HorseCheck,
    RookCheck,
    CannonCheck,
    SoldierCheck,
}

impl MatingPattern {
    pub const ALL: [Self; 18] = [
        Self::DoubleCannon,
        Self::HorseCannon,
        Self::IronBolt,
        Self::SmotheredCannon,
        Self::HeavenEarthCannon,
        Self::TwoSoldiers,
        Self::WhiteFacedGeneral,
        Self::ElbowHorse,
        Self::PalaceCornerHorse,
        Self::AnglerHorse,
        Self::OctagonalHorse,
        Self::DoubleRook,
        Self::SmotheredMate,
        Self::DoubleCheck,
        Self::HorseCheck,
        Self::RookCheck,
        Self::CannonCheck,
        Self::SoldierCheck,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::DoubleCannon => "重炮",
            Self::HorseCannon => "马后炮",
            Self::IronBolt => "铁门栓",
            Self::SmotheredCannon => "闷宫",
            Self::HeavenEarthCannon => "天地炮",
            Self::TwoSoldiers => "二鬼拍门",
            Self::WhiteFacedGeneral => "白脸将",
            Self::ElbowHorse => "卧槽马",
            Self::PalaceCornerHorse => "挂角马",
            Self::AnglerHorse => "钓鱼马",
            Self::OctagonalHorse => "八角马",
            Self::DoubleRook => "双车配合",
            Self::SmotheredMate => "闷杀（终局形）",
            Self::DoubleCheck => "双将",
            Self::HorseCheck => "马将死",
            Self::RookCheck => "车将死",
            Self::CannonCheck => "炮将死",
            Self::SoldierCheck => "兵卒将死",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MateClassification {
    pub winner: Color,
    pub kind: MateKind,
    /// 按 MatingPattern::ALL 顺序返回，无重复；无匹配杀形时可为空。
    pub patterns: Vec<MatingPattern>,
    /// 实际正在将军的棋子格号，升序。支持子不在此列。
    pub checkers: Vec<usize>,
}

impl Position {
    /// 识别轮到走棋一方的棋盘终局。非终局、缺将或进攻方也被将军返回 None。
    /// 困毙返回独立的 Stalemate，杀形标签为空。
    /// 不判断历史规则终局；调用方应先处理对局历史的胜负/和棋。
    pub fn classify_mate(&self) -> Option<MateClassification> {
        let defender = self.side_to_move();
        let winner = defender.opposite();
        let king = self.general_square(defender)?;
        let attacking_king = self.general_square(winner)?;
        if !inside_palace(defender, file_of(king), rank_of(king))
            || !inside_palace(winner, file_of(attacking_king), rank_of(attacking_king))
            || self.in_check(winner)
            || !self.legal_moves().is_empty()
        {
            return None;
        }
        let checkers = attackers(self, king, winner);
        let mut result = MateClassification {
            winner,
            kind: if checkers.is_empty() {
                MateKind::Stalemate
            } else {
                MateKind::Checkmate
            },
            patterns: Vec::new(),
            checkers,
        };
        if result.kind == MateKind::Stalemate {
            return Some(result);
        }
        let context = MateContext {
            position: self,
            king,
            winner,
            checkers: &result.checkers,
        };
        result.patterns = MatingPattern::ALL
            .into_iter()
            .filter(|&p| context.matches(p))
            .collect();
        Some(result)
    }
}

fn attackers(position: &Position, target: usize, by: Color) -> Vec<usize> {
    let mut squares = Vec::new();
    position.visit_attacker_origins_to(target, by, |from| {
        squares.push(from);
        false
    });
    squares.sort_unstable();
    squares
}

struct MateContext<'a> {
    position: &'a Position,
    king: usize,
    winner: Color,
    checkers: &'a [usize],
}

impl MateContext<'_> {
    /// 统一为防守方底线 rank=0，向棋盘中部递增的坐标。
    fn rank(&self, square: usize) -> usize {
        if self.winner == Color::Red {
            rank_of(square)
        } else {
            9 - rank_of(square)
        }
    }

    fn square(&self, file: usize, rank: usize) -> usize {
        index(
            file,
            if self.winner == Color::Red {
                rank
            } else {
                9 - rank
            },
        )
    }

    fn kind(&self, square: usize) -> PieceKind {
        self.position
            .piece_at(square)
            .expect("occupied mating participant")
            .kind
    }

    fn has(&self, square: usize, kind: PieceKind, color: Color) -> bool {
        self.position
            .piece_at(square)
            .is_some_and(|p| p.color == color && p.kind == kind)
    }

    fn pieces(&self, kind: PieceKind) -> impl Iterator<Item = usize> + '_ {
        self.position
            .occupied_squares()
            .filter(move |&sq| self.has(sq, kind, self.winner))
    }

    fn checking(&self, kind: PieceKind) -> impl Iterator<Item = usize> + '_ {
        self.checkers
            .iter()
            .copied()
            .filter(move |&sq| self.kind(sq) == kind)
    }

    fn screen(&self, cannon: usize) -> Option<usize> {
        let mut occupied = line_between_squares(cannon, self.king)
            .into_iter()
            .filter(|&sq| self.position.piece_at(sq).is_some());
        let screen = occupied.next()?;
        occupied.next().is_none().then_some(screen)
    }

    /// 不删子、不改炮架：实际执行宫内将步后，只有指定支持子阻止此步时才算贡献。
    fn denies_escape(&self, supporter: usize) -> bool {
        self.position
            .pseudo_legal_moves()
            .into_iter()
            .filter(|mv| mv.from as usize == self.king)
            .any(|mv| self.only_attack_after(mv, supporter))
    }

    fn only_attack_after(&self, mv: Move, source: usize) -> bool {
        let mut next = self.position.clone();
        next.make_move(mv);
        let king = next
            .general_square(self.winner.opposite())
            .expect("general moved, not captured");
        attackers(&next, king, self.winner) == [source]
    }

    /// 中炮当前不将军，但牵制两颗中路防守子。忽略该炮攻击后，原局面的一步
    /// 伪合法守子应着可以解将；所有占位保持真实，避免移除炮导致虚假通路。
    fn central_pin(&self, cannon: usize) -> bool {
        if file_of(self.king) != 4 || file_of(cannon) != 4 || self.checkers.contains(&cannon) {
            return false;
        }
        let between = line_between_squares(cannon, self.king);
        if between
            .iter()
            .filter(|&&sq| self.position.piece_at(sq).is_some())
            .count()
            != 2
        {
            return false;
        }
        self.position.pseudo_legal_moves().into_iter().any(|mv| {
            let from = mv.from as usize;
            from != self.king && between.contains(&from) && self.only_attack_after(mv, cannon)
        })
    }

    fn matches(&self, pattern: MatingPattern) -> bool {
        use MatingPattern::*;
        match pattern {
            DoubleCannon => self.checking(PieceKind::Cannon).any(|sq| {
                self.screen(sq)
                    .is_some_and(|s| self.has(s, PieceKind::Cannon, self.winner))
            }),
            HorseCannon => self.checking(PieceKind::Cannon).any(|sq| {
                self.screen(sq).is_some_and(|s| {
                    self.has(s, PieceKind::Horse, self.winner)
                        && file_of(s).abs_diff(file_of(self.king))
                            + rank_of(s).abs_diff(rank_of(self.king))
                            == 2
                        && self.denies_escape(s)
                })
            }),
            IronBolt => {
                self.checkers.iter().any(|&sq| {
                    matches!(self.kind(sq), PieceKind::Rook | PieceKind::Soldier)
                        && ((self.rank(sq) == 0 && self.rank(self.king) == 0)
                            || matches!(file_of(sq), 3 | 5))
                }) && self
                    .pieces(PieceKind::Cannon)
                    .any(|sq| self.central_pin(sq))
            }
            SmotheredCannon => {
                self.rank(self.king) == 0
                    && file_of(self.king) == 4
                    && self.has(
                        self.square(4, 1),
                        PieceKind::Advisor,
                        self.winner.opposite(),
                    )
                    && self.checking(PieceKind::Cannon).any(|sq| {
                        self.rank(sq) == 0
                            && self.screen(sq).is_some_and(|s| {
                                matches!(file_of(s), 3 | 5)
                                    && self.has(s, PieceKind::Advisor, self.winner.opposite())
                            })
                    })
            }
            HeavenEarthCannon => {
                self.checking(PieceKind::Cannon)
                    .any(|sq| self.rank(sq) == 0)
                    && self
                        .pieces(PieceKind::Cannon)
                        .any(|sq| self.central_pin(sq))
            }
            TwoSoldiers => self.checking(PieceKind::Soldier).any(|checker| {
                matches!(file_of(checker), 3 | 5)
                    && self.rank(checker) <= 1
                    && self.pieces(PieceKind::Soldier).any(|support| {
                        file_of(support) == 8 - file_of(checker)
                            && self.rank(support) <= 1
                            && self.denies_escape(support)
                    })
            }),
            WhiteFacedGeneral => {
                self.denies_escape(self.position.general_square(self.winner).unwrap())
            }
            ElbowHorse => self
                .checking(PieceKind::Horse)
                .any(|sq| matches!(file_of(sq), 2 | 6) && self.rank(sq) == 1),
            PalaceCornerHorse => self
                .checking(PieceKind::Horse)
                .any(|sq| matches!(file_of(sq), 3 | 5) && self.rank(sq) == 2),
            AnglerHorse => self.pieces(PieceKind::Horse).any(|sq| {
                matches!(file_of(sq), 2 | 6)
                    && self.rank(sq) == 2
                    && !self.checkers.contains(&sq)
                    && self.denies_escape(sq)
            }),
            OctagonalHorse => self.pieces(PieceKind::Horse).any(|sq| {
                matches!(file_of(sq), 3 | 5)
                    && self.rank(sq) == 2
                    && !self.checkers.contains(&sq)
                    && self.denies_escape(sq)
            }),
            DoubleRook => {
                self.checking(PieceKind::Rook).next().is_some()
                    && self
                        .pieces(PieceKind::Rook)
                        .any(|sq| !self.checkers.contains(&sq) && self.denies_escape(sq))
            }
            SmotheredMate => super::ORTHOGONAL_STEPS.into_iter().all(|(df, dr)| {
                let f = file_of(self.king) as i32 + df;
                let r = rank_of(self.king) as i32 + dr;
                !super::inside_board(f, r)
                    || !inside_palace(self.winner.opposite(), f as usize, r as usize)
                    || self
                        .position
                        .piece_at(index(f as usize, r as usize))
                        .is_some_and(|p| p.color == self.winner.opposite())
            }),
            DoubleCheck => self.checkers.len() >= 2,
            HorseCheck => self.checking(PieceKind::Horse).next().is_some(),
            RookCheck => self.checking(PieceKind::Rook).next().is_some(),
            CannonCheck => self.checking(PieceKind::Cannon).next().is_some(),
            SoldierCheck => self.checking(PieceKind::Soldier).next().is_some(),
        }
    }
}
