use crate::az::nnue::{AZ_NNUE_INPUT_SIZE, V2_KING_BUCKETS};
use crate::xiangqi::{BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move};

use super::px0_policy_map;

pub(crate) const SPARSE_MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
pub const DENSE_MOVE_SPACE: usize = 2062;
pub(crate) const POLICY_CONSEQUENCE_SIZE: usize = 32;
pub(crate) const POLICY_MOVE_CONTEXT_SIZE: usize = 16;
pub(crate) const POLICY_THREAT_CONTEXT_SIZE: usize = 16;
pub(crate) const POLICY_ACCUMULATOR_RANK: usize = 64;
pub(crate) const POLICY_TACTICAL_SIGNATURE_BUCKETS: usize = 64;
pub(crate) const POLICY_TACTICAL_TERMS: usize = 3;
pub const POLICY_TACTICAL_EXACT_SIZE: usize =
    DENSE_MOVE_SPACE * (STRUCTURAL_PIECE_SIZE / 2) * POLICY_TACTICAL_SIGNATURE_BUCKETS;
pub const POLICY_TACTICAL_FACTOR_SIZE: usize =
    (STRUCTURAL_PIECE_SIZE / 2) * POLICY_TACTICAL_SIGNATURE_BUCKETS;
pub(crate) const POLICY_CAPTURE_RELATION_OFFSET: usize =
    POLICY_TACTICAL_EXACT_SIZE + POLICY_TACTICAL_FACTOR_SIZE;
pub(crate) const POLICY_CAPTURE_RELATION_BUCKETS: usize = 32;
pub(crate) const POLICY_CAPTURE_RELATION_SIZE: usize = 7 * 7 * POLICY_CAPTURE_RELATION_BUCKETS;
pub(crate) const POLICY_TACTICAL_SIZE: usize =
    POLICY_CAPTURE_RELATION_OFFSET + POLICY_CAPTURE_RELATION_SIZE;
// 推理只访问己方走子和敌方被吃子；训练张量保持原布局。
pub(crate) const POLICY_CACHE_PIECE_SIZE: usize = STRUCTURAL_PIECE_SIZE / 2;
pub(crate) const POLICY_CACHE_CAPTURE_CLASSES: usize = POLICY_CACHE_PIECE_SIZE + 1;
pub(crate) const POLICY_CACHE_MAIN_SIZE: usize =
    DENSE_MOVE_SPACE * POLICY_CACHE_PIECE_SIZE * V2_KING_BUCKETS * V2_KING_BUCKETS;
pub(crate) const POLICY_CACHE_TABLE_SIZE: usize =
    POLICY_CACHE_MAIN_SIZE + DENSE_MOVE_SPACE * POLICY_CACHE_CAPTURE_CLASSES;
pub(crate) const POLICY_SPARSE_CAPTURE_CLASSES: usize = STRUCTURAL_PIECE_SIZE + 1;
pub const POLICY_SPARSE_MAIN_SIZE: usize =
    DENSE_MOVE_SPACE * STRUCTURAL_PIECE_SIZE * V2_KING_BUCKETS * V2_KING_BUCKETS;
pub(crate) const POLICY_SPARSE_CAPTURE_SIZE: usize =
    DENSE_MOVE_SPACE * POLICY_SPARSE_CAPTURE_CLASSES;
pub(crate) const POLICY_SPARSE_TABLE_SIZE: usize =
    POLICY_SPARSE_MAIN_SIZE + POLICY_SPARSE_CAPTURE_SIZE + 1;
pub(crate) const POLICY_SPARSE_MOVE_PIECE_SIZE: usize = DENSE_MOVE_SPACE * STRUCTURAL_PIECE_SIZE;
pub(crate) const POLICY_SPARSE_MOVE_KING_SIZE: usize =
    DENSE_MOVE_SPACE * V2_KING_BUCKETS * V2_KING_BUCKETS;
pub(crate) const POLICY_SPARSE_PIECE_KING_SIZE: usize =
    STRUCTURAL_PIECE_SIZE * V2_KING_BUCKETS * V2_KING_BUCKETS;
pub(crate) const POLICY_KING_DISTANCE_BUCKETS: usize = 6;
pub(crate) const POLICY_KING_APPROACH_BUCKETS: usize = 5;
pub(crate) const POLICY_SPARSE_DISTANCE_SIZE: usize =
    STRUCTURAL_PIECE_SIZE * POLICY_KING_DISTANCE_BUCKETS;
pub(crate) const POLICY_SPARSE_APPROACH_SIZE: usize =
    STRUCTURAL_PIECE_SIZE * POLICY_KING_APPROACH_BUCKETS;
pub(crate) const POLICY_SPARSE_FACTOR_SIZE: usize = POLICY_SPARSE_MOVE_PIECE_SIZE
    + POLICY_SPARSE_MOVE_KING_SIZE
    + POLICY_SPARSE_PIECE_KING_SIZE
    + POLICY_SPARSE_DISTANCE_SIZE
    + POLICY_SPARSE_APPROACH_SIZE;
pub(crate) const POLICY_ACCUMULATOR_PIECE_OFFSET: usize = AZ_NNUE_INPUT_SIZE;
pub(crate) const POLICY_ACCUMULATOR_RANK_OFFSET: usize =
    POLICY_ACCUMULATOR_PIECE_OFFSET + STRUCTURAL_PIECE_SIZE;
pub(crate) const POLICY_ACCUMULATOR_FILE_OFFSET: usize = POLICY_ACCUMULATOR_RANK_OFFSET + STRUCTURAL_RANK_SIZE;
pub(crate) const POLICY_ACCUMULATOR_KING_PIECE_OFFSET: usize =
    POLICY_ACCUMULATOR_FILE_OFFSET + STRUCTURAL_FILE_SIZE;
pub(crate) const POLICY_ACCUMULATOR_BIAS_ROW: usize =
    POLICY_ACCUMULATOR_KING_PIECE_OFFSET + STRUCTURAL_KING_PIECE_SIZE;
pub(crate) const POLICY_ACCUMULATOR_ROWS: usize = POLICY_ACCUMULATOR_BIAS_ROW + 1;
pub(crate) const VALUE_HEAD_SIZE: usize = 96;
pub(crate) const VALUE_KING_PIECE_VOCAB: usize = 2 * V2_KING_BUCKETS * 14 * BOARD_SIZE;
pub(crate) const VALUE_KING_PIECE_MAX_ACTIVE: usize = 64;
pub(crate) const VALUE_THREAT_RANK: usize = 64;
pub(crate) const VALUE_THREAT_PAIR_VOCAB: usize = 57_702;
pub(crate) const VALUE_RAY_VOCAB: usize = 4 * 2 * 4 * 9 * 15 * 4;
pub(crate) const VALUE_CANNON_TRIPLE_VOCAB: usize = 32_768;
pub(crate) const VALUE_THREAT_VOCAB: usize =
    VALUE_THREAT_PAIR_VOCAB + VALUE_RAY_VOCAB + VALUE_CANNON_TRIPLE_VOCAB;
pub(crate) const VALUE_THREAT_MAX_ACTIVE: usize = 192;
pub(crate) const WDL_HEAD_SIZE: usize = 3;
/// Small, exact-history-derived signals.  These deliberately replace the old
/// high-dimensional history planes: rules stay in the environment, while the
/// network only gets enough context to recognize an approaching repetition.
pub const RULE_CONTEXT_SIZE: usize = 7;
#[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
pub(crate) const RMS_NORM_EPS: f32 = 1.0e-6;
pub(crate) const PIECE_SQUARE_INPUT_SIZE: usize = BOARD_SIZE * 14;
pub(crate) const STRUCTURAL_PIECE_SIZE: usize = 14;
pub(crate) const STRUCTURAL_RANK_SIZE: usize = BOARD_RANKS;
pub(crate) const STRUCTURAL_FILE_SIZE: usize = BOARD_FILES;
pub(crate) const STRUCTURAL_KING_PIECE_SIZE: usize = 2 * V2_KING_BUCKETS * 14;

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

pub(crate) fn canonical_general_buckets_from_features(features: &[usize]) -> (usize, usize) {
    let mut us = 4;
    let mut them = 4;
    for &feature in features {
        if feature >= PIECE_SQUARE_INPUT_SIZE {
            continue;
        }
        let piece_index = feature / BOARD_SIZE;
        let sq = feature % BOARD_SIZE;
        match piece_index {
            0 => us = canonical_general_bucket(piece_index, sq),
            7 => them = canonical_general_bucket(piece_index, sq),
            _ => {}
        }
    }
    (us, them)
}

pub(crate) fn structural_king_piece_index(
    perspective: usize,
    king_bucket: usize,
    piece_index: usize,
) -> usize {
    ((perspective * V2_KING_BUCKETS + king_bucket.min(V2_KING_BUCKETS - 1)) * 14) + piece_index
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
        Self { hidden_size: 128 }
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

#[inline(always)]
pub(crate) fn canonical_square_for(perspective: Color, sq: usize) -> usize {
    if perspective == Color::Red {
        sq
    } else {
        BOARD_SIZE - 1 - sq
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
