//! 终局杀形的多标签识别。定义、来源和刻意限制见 docs/mating-patterns.md。
//! 只使用棋盘合法着法，不把长将判负、搜索证明或步数截断当成将死。

use super::{
    Color, Move, PieceKind, Position, file_of, inside_palace, line_between_squares, rank_of,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MateKind {
    Checkmate,
    Stalemate,
}

/// 仅保留已确认的 11 种具名杀法；同一终局可以命中多种。
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MatingPattern {
    DoubleCannon,
    IronBolt,
    HorseCannon,
    WhiteFacedGeneral,
    ElbowHorse,
    PalaceCornerHorse,
    AnglerHorse,
    OctagonalHorse,
    HighAnglerHorse,
    DoubleHorse,
    CrossRooks,
}

impl MatingPattern {
    pub const ALL: [Self; 11] = [
        Self::DoubleCannon,
        Self::IronBolt,
        Self::HorseCannon,
        Self::WhiteFacedGeneral,
        Self::ElbowHorse,
        Self::PalaceCornerHorse,
        Self::AnglerHorse,
        Self::OctagonalHorse,
        Self::HighAnglerHorse,
        Self::DoubleHorse,
        Self::CrossRooks,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::DoubleCannon => "重炮",
            Self::HorseCannon => "马后炮",
            Self::IronBolt => "铁门栓",
            Self::WhiteFacedGeneral => "白脸将",
            Self::ElbowHorse => "卧槽马",
            Self::PalaceCornerHorse => "挂角马",
            Self::AnglerHorse => "钓鱼马",
            Self::OctagonalHorse => "八角马",
            Self::HighAnglerHorse => "高钓马",
            Self::DoubleHorse => "双马饮泉",
            Self::CrossRooks => "双车错",
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
                    && matches!(self.rank(sq), 0 | 2)
                    && self.rank(self.king) == 2 - self.rank(sq)
                    && file_of(self.king) == 8 - file_of(sq)
                    && !self.checkers.contains(&sq)
                    && self.denies_escape(sq)
            }),
            HighAnglerHorse => self.pieces(PieceKind::Horse).any(|sq| {
                matches!(file_of(sq), 2 | 6)
                    && self.rank(sq) == 3
                    && !self.checkers.contains(&sq)
                    && self.denies_escape(sq)
            }),
            DoubleHorse => self.checking(PieceKind::Horse).any(|checker| {
                self.pieces(PieceKind::Horse).any(|support| {
                    support != checker
                        && (self.checkers.contains(&support) || self.denies_escape(support))
                })
            }),
            // 识别双车交错两线的最终杀形，不声称证明此前反复交替将军的过程。
            CrossRooks => self.checking(PieceKind::Rook).any(|checker| {
                self.pieces(PieceKind::Rook).any(|support| {
                    !self.checkers.contains(&support)
                        && file_of(checker) != file_of(support)
                        && rank_of(checker) != rank_of(support)
                        && self.denies_escape(support)
                })
            }),
        }
    }
}
