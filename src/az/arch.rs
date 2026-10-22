use super::px0_policy_map;
use crate::xiangqi::{BOARD_FILES, BOARD_SIZE, Move};
pub(crate) const SPARSE_MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
pub const DENSE_MOVE_SPACE: usize = 2062;
pub const RULE_CONTEXT_SIZE: usize = 18;
pub const CHECK_CONTEXT_SIZE: usize = 14;
pub(crate) const PIECE_SQUARE_INPUT_SIZE: usize = 1260;
pub(crate) const VALUE_HEAD_SIZE: usize = 96;
pub(crate) const VALUE_BUCKETS: usize = 32;
pub(crate) const VALUE_KING_PIECE_RANK: usize = 16;
pub(crate) const VALUE_KING_PIECE_VOCAB: usize = 22680;
pub(crate) const WDL_HEAD_SIZE: usize = 3;
pub(crate) const POLICY_MOVE_CONTEXT_SIZE: usize = 16;
#[derive(Clone, Copy, Debug)]
pub(crate) struct StructuralPieceSquare {
    pub piece_index: usize,
    pub rank: usize,
    pub file: usize,
}

pub(crate) fn decode_current_piece_square_feature(feature: usize) -> Option<StructuralPieceSquare> {
    if feature >= PIECE_SQUARE_INPUT_SIZE {
        return None;
    }
    let piece_index = feature / BOARD_SIZE;
    let sq = feature % BOARD_SIZE;
    Some(StructuralPieceSquare {
        piece_index,
        rank: sq / BOARD_FILES,
        file: sq % BOARD_FILES,
    })
}

pub(crate) fn canonical_general_bucket(piece_index: usize, sq: usize) -> usize {
    let oriented_sq = if piece_index < 7 {
        sq
    } else {
        BOARD_SIZE - 1 - sq
    };
    let file = (oriented_sq % BOARD_FILES).clamp(3, 5) - 3;
    let rank = (oriented_sq / BOARD_FILES).clamp(7, 9) - 7;
    rank * 3 + file
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AzNnueArch {
    pub hidden_size: usize,
}

impl AzNnueArch {
    pub const fn default_const() -> Self {
        Self { hidden_size: 96 }
    }

    pub const fn with_hidden_size(hidden_size: usize) -> Self {
        let mut arch = Self::default_const();
        arch.hidden_size = hidden_size;
        arch
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err(format!("invalid hidden_size {}", self.hidden_size));
        }
        Ok(())
    }
}

impl Default for AzNnueArch {
    fn default() -> Self {
        Self::default_const()
    }
}

pub(crate) struct MoveMap {
    pub(crate) sparse_to_dense: [u16; SPARSE_MOVE_SPACE],
    #[allow(dead_code)]
    pub(crate) dense_to_sparse: [u16; DENSE_MOVE_SPACE],
}

pub(crate) fn move_map() -> &'static MoveMap {
    use std::sync::OnceLock;
    static MAP: OnceLock<MoveMap> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut sparse_to_dense = [u16::MAX; SPARSE_MOVE_SPACE];
        let dense_to_sparse = px0_policy_map::PX0_MOVES;
        for (index, &sparse) in dense_to_sparse.iter().enumerate() {
            assert_eq!(sparse_to_dense[sparse as usize], u16::MAX);
            sparse_to_dense[sparse as usize] = index as u16;
        }
        MoveMap {
            sparse_to_dense,
            dense_to_sparse,
        }
    })
}

pub(crate) fn dense_move_squares(move_index: usize) -> Option<(usize, usize)> {
    let sparse = *move_map().dense_to_sparse.get(move_index)? as usize;
    Some((sparse / BOARD_SIZE, sparse % BOARD_SIZE))
}

pub fn dense_move_index(mv: Move) -> usize {
    let sparse = mv.from as usize * BOARD_SIZE + mv.to as usize;
    let dense = move_map().sparse_to_dense[sparse];
    debug_assert!(
        dense != u16::MAX,
        "invalid policy move {}->{}",
        mv.from,
        mv.to
    );
    dense as usize
}

/// MLH 使用半回合；归一化用于稳定回归梯度。
pub(crate) const MOVES_LEFT_SCALE: f32 = 100.0;
pub(crate) const MOVES_LEFT_LOSS_WEIGHT: f32 = 0.05;
/// 剩余步数对搜索的影响；默认参数来自 PX0/LC0 classic 搜索。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AzMovesLeftParams {
    pub enabled: bool,
    pub threshold: f32,
    pub max_effect: f32,
    pub slope: f32,
    pub constant_factor: f32,
    pub scaled_factor: f32,
    pub quadratic_factor: f32,
}

impl Default for AzMovesLeftParams {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.8,
            max_effect: 0.0345,
            slope: 0.0027,
            constant_factor: 0.0,
            scaled_factor: 1.6521,
            quadratic_factor: -0.6521,
        }
    }
}

impl AzMovesLeftParams {
    pub fn normalize(mut self) -> Self {
        fn bounded(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
            if value.is_finite() {
                value.clamp(min, max)
            } else {
                fallback
            }
        }
        let defaults = Self::default();
        self.threshold = bounded(self.threshold, 0.0, 1.0, defaults.threshold);
        self.max_effect = bounded(self.max_effect, 0.0, 1.0, defaults.max_effect);
        self.slope = bounded(self.slope, 0.0, 1.0, defaults.slope);
        self.constant_factor = bounded(self.constant_factor, -1.0, 1.0, defaults.constant_factor);
        self.scaled_factor = bounded(self.scaled_factor, -2.0, 2.0, defaults.scaled_factor);
        self.quadratic_factor =
            bounded(self.quadratic_factor, -1.0, 1.0, defaults.quadratic_factor);
        self
    }
}
