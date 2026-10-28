//! 策略头：局面与走法交互、候选先验表和重复修正。
use super::*;
use crate::az::nnue::{V2_KING_BUCKETS, piece_absolute_feature_index};
use crate::xiangqi::{BOARD_SIZE, Piece, piece_kind_index};

#[inline]
pub(crate) fn policy_move_pieces(
    position: &Position,
    side: Color,
    mv: Move,
) -> Option<(usize, Option<usize>)> {
    let moved = piece_absolute_feature_index(side, position.piece_at(mv.from as usize)?);
    let captured = position
        .piece_at(mv.to as usize)
        .map(|piece| piece_absolute_feature_index(side, piece));
    Some((moved, captured))
}

#[inline]
pub(crate) fn policy_sparse_main_index(
    move_index: usize,
    moved_piece: usize,
    us_king_bucket: usize,
    them_king_bucket: usize,
) -> usize {
    debug_assert!(move_index < DENSE_MOVE_SPACE && moved_piece < POLICY_PIECE_SIZE);
    debug_assert!(us_king_bucket < V2_KING_BUCKETS && them_king_bucket < V2_KING_BUCKETS);
    policy_main_compact_index(move_index, moved_piece, us_king_bucket, them_king_bucket)
        .expect("策略表不支持该走法、棋种与双将位组合")
}

#[inline]
pub(crate) fn policy_sparse_capture_index(
    move_index: usize,
    captured_piece: Option<usize>,
) -> usize {
    policy_capture_compact_index(move_index, captured_piece)
        .expect("吃子表不支持该走法与被吃棋种组合")
}

#[inline]
pub(crate) fn policy_tactical_indices(
    move_index: usize,
    moved_piece: usize,
    source_attacked: bool,
    destination_attacked: bool,
    source_defended: bool,
    destination_defended: bool,
    captured_piece: Option<usize>,
    check: bool,
) -> [usize; POLICY_TACTICAL_TERMS] {
    debug_assert!(moved_piece < STRUCTURAL_PIECE_SIZE / 2);
    let signature = policy_tactical_signature(
        source_attacked,
        destination_attacked,
        source_defended,
        destination_defended,
        captured_piece.is_some(),
        check,
    );
    let exact = policy_tactical_exact_index(move_index, moved_piece, signature)
        .expect("战术签名不满足走前攻击位板约束");
    let relation = captured_piece.map_or(POLICY_TACTICAL_SIZE, |victim| {
        policy_capture_relation_index(moved_piece, victim, signature)
            .expect("吃子关系表不支持该棋种配对")
    });
    [exact, relation]
}

pub(crate) fn policy_tactical_signature(
    source_attacked: bool,
    destination_attacked: bool,
    source_defended: bool,
    destination_defended: bool,
    capture: bool,
    check: bool,
) -> usize {
    usize::from(source_attacked)
        | usize::from(destination_attacked) << 1
        | usize::from(source_defended) << 2
        | usize::from(destination_defended) << 3
        | usize::from(capture) << 4
        | usize::from(check) << 5
}

/// 策略头战术签名用到的 4 个布尔量，返回顺序与 `policy_tactical_indices` 的入参一致：
/// `(source_attacked, destination_attacked, source_defended, destination_defended)`。
///
/// 四个位都用**走前**攻击位板，语义要按"现在"读，不要按"走后"读：
/// - `source_attacked/defended` = 我此刻站着的这格是否被攻击/被保护；
/// - `destination_attacked/defended` = **此刻**落点这一格是否被攻击/被保护。
///
/// 注意第二组**不是**"落子之后那枚子是否被攻击/被保护"：两者实测约 **21%** 不一致，
/// 而且差异是系统性的、不是罕见边角——沿直线走子时走子方自己就攻击着落点（车/炮/兵
/// 沿线移动），于是走前 `defended` 必然为真，走后却只取决于有没有**别的**子保护它；
/// 此外 `from` 腾空还会让炮失去炮架、让车线打开、松开马腿/象眼。
///
/// 明知有约 21% 的语义差仍用近似，是因为精确值要按每个候选走法各查两次，实测
/// **−25%~−39% NPS**（startpos 201k → 123k、中局 111k → 83k sims/s）。训练侧与推理侧
/// 共用这一个实现，所以模型学到的就是"我们控制这一格"这个自洽信号。精确查询留在
/// `Position::is_square_attacked_after_move`，只作审计/测试口径
/// （见 `tactical_flags_are_pre_move_and_the_gap_is_audited`），生产路径不调用它。
#[inline]
pub(crate) fn policy_move_tactical_flags(
    mv: Move,
    opponent_attacks: u128,
    own_attacks: u128,
) -> (bool, bool, bool, bool) {
    let from = mv.from as usize;
    let to = mv.to as usize;
    let source_attacked = opponent_attacks & (1u128 << from) != 0;
    let source_defended = own_attacks & (1u128 << from) != 0;
    let destination_attacked = opponent_attacks & (1u128 << to) != 0;
    let destination_defended = own_attacks & (1u128 << to) != 0;
    (
        source_attacked,
        destination_attacked,
        source_defended,
        destination_defended,
    )
}

impl AzNnue {
    pub(crate) fn evaluate_policy_with_scratch(
        &self,
        position: &Position,
        moves: &[Move],
        repetition_flags: &[u8],
        scratch: &mut AzEvalScratch,
    ) {
        scratch.policy_context.resize(POLICY_MOVE_CONTEXT_SIZE, 0.0);
        for (context_index, context) in scratch.policy_context.iter_mut().enumerate() {
            let start = context_index * self.hidden_size;
            *context = dot_product(
                &scratch.hidden,
                &self.policy_context_hidden[start..start + self.hidden_size],
            );
            if self.threat_active {
                let threat_start = context_index * VALUE_THREAT_RANK * 2;
                *context += dot_product(
                    &scratch.threat_activation,
                    &self.policy_threat_context[threat_start..threat_start + VALUE_THREAT_RANK * 2],
                );
            }
        }
        scratch.logits.resize(moves.len(), 0.0);
        let move_map = move_map();
        let side = position.side_to_move();
        let king_buckets = canonical_buckets_for_perspective(position, side);
        // 主干之前如果已经算过（标量块需要），这里直接复用，避免同一个局面算两遍：
        // 提前算的总工作量不变，只是把它挪到主干之前。
        if !scratch.policy_inputs_ready {
            self.fill_policy_inputs(position, moves, scratch, self.policy_tactical_active);
        }
        let attack_masks = if self.policy_tactical_active {
            scratch.attack_masks
        } else {
            [0u128; 2]
        };
        let opponent_attacks = attack_masks[color_index(side.opposite())];
        let own_attacks = attack_masks[color_index(side)];
        let repetition_logit = dot_product(&scratch.hidden, &self.policy_repetition_hidden)
            + self.policy_repetition_bias[0];
        {
            crate::scope_profile!("az.eval.policy_logits");
            {
                crate::scope_profile!("az.eval.policy.logit_arith");
                for (index, mv) in moves.iter().enumerate() {
                    let canonical = canonical_move(side, *mv);
                    let sparse = canonical.from as usize * BOARD_SIZE + canonical.to as usize;
                    let dense = move_map.sparse_to_dense[sparse];
                    debug_assert!(
                        dense != u16::MAX,
                        "invalid policy move {}->{}",
                        mv.from,
                        mv.to
                    );
                    let move_index = dense as usize;
                    let context_start = move_index * POLICY_MOVE_CONTEXT_SIZE;
                    let accumulator_start = move_index * POLICY_ACCUMULATOR_RANK;
                    let accumulator_move = &self.policy_accumulator_move
                        [accumulator_start..accumulator_start + POLICY_ACCUMULATOR_RANK];
                    let pieces = policy_move_pieces(position, side, *mv);
                    let accumulator_logit = pieces.map_or(0.0, |_| {
                        dot_product(&scratch.policy_accumulator_context, accumulator_move)
                    });
                    let prior = pieces.map_or(0.0, |(moved, captured)| {
                        let signature = self.policy_tactical_active.then(|| {
                            let (sa, da, sd, dd) =
                                policy_move_tactical_flags(*mv, opponent_attacks, own_attacks);
                            policy_tactical_signature(
                                sa,
                                da,
                                sd,
                                dd,
                                captured.is_some(),
                                scratch.policy_gives_check[index] != 0.0,
                            )
                        });
                        self.policy_prior_logit(
                            move_index,
                            moved,
                            captured,
                            king_buckets,
                            signature,
                        )
                    });
                    scratch.logits[index] = dot_product(
                        &scratch.policy_context,
                        &self.policy_move_context
                            [context_start..context_start + POLICY_MOVE_CONTEXT_SIZE],
                    ) + accumulator_logit
                        + prior
                        + f32::from(repetition_flags.get(index).copied().unwrap_or(0))
                            * repetition_logit;
                }
            }
        }
    }

    /// 静态候选先验独立于动态局面交互；所有紧凑表在此汇总。
    fn policy_prior_logit(
        &self,
        mv: usize,
        moved: usize,
        captured: Option<usize>,
        kings: (usize, usize),
        signature: Option<usize>,
    ) -> f32 {
        let main = policy_sparse_main_index(mv, moved, kings.0, kings.1);
        let capture = policy_sparse_capture_index(mv, captured);
        let mut prior = self.policy_sparse_table[main] + self.policy_sparse_table[capture];
        if let Some(signature) = signature {
            let row = policy_tactical_exact_index(mv, moved, signature)
                .expect("战术签名不满足候选走法约束");
            prior += self.policy_tactical[row];
            if let Some(victim) = captured {
                let row = policy_capture_relation_index(moved, victim, signature)
                    .expect("吃子关系表不支持该棋种配对");
                prior += self.policy_tactical[row];
            }
        }
        prior
    }

    pub(crate) fn rebuild_policy_cache(&mut self) {
        let mut projected = Vec::with_capacity(POLICY_ACCUMULATOR_ROWS * POLICY_ACCUMULATOR_RANK);
        let projection = &self.policy_accumulator_hidden;
        let hidden = self.hidden_size;
        let mut append = |table: &[f32]| {
            debug_assert_eq!(table.len() % hidden, 0);
            for row in table.chunks_exact(hidden) {
                for rank in 0..POLICY_ACCUMULATOR_RANK {
                    let weights = &projection[rank * hidden..(rank + 1) * hidden];
                    projected.push(dot_product(row, weights));
                }
            }
        };
        append(&self.input_hidden);
        append(&self.hidden_bias);
        debug_assert_eq!(
            projected.len(),
            POLICY_ACCUMULATOR_ROWS * POLICY_ACCUMULATOR_RANK
        );
        self.policy_accumulator_features = projected;
    }

    pub(crate) fn policy_accumulator(
        &self,
        position: &Position,
        perspective: Color,
    ) -> [f32; POLICY_ACCUMULATOR_RANK] {
        let mut accumulator = [0.0; POLICY_ACCUMULATOR_RANK];
        self.add_policy_row(&mut accumulator, POLICY_ACCUMULATOR_BIAS_ROW, 1);
        for square in 0..BOARD_SIZE {
            if let Some(piece) = position.piece_at(square) {
                self.add_policy_piece(&mut accumulator, perspective, square, piece, 1);
            }
        }
        accumulator
    }

    pub(crate) fn apply_policy_transition(
        &self,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        perspective: Color,
        accumulator: &mut [f32; POLICY_ACCUMULATOR_RANK],
    ) {
        self.add_policy_piece(accumulator, perspective, mv.from as usize, moved, -1);
        if let Some(captured) = captured {
            self.add_policy_piece(accumulator, perspective, mv.to as usize, captured, -1);
        }
        self.add_policy_piece(accumulator, perspective, mv.to as usize, moved, 1);
    }

    pub(crate) fn add_policy_piece(
        &self,
        accumulator: &mut [f32; POLICY_ACCUMULATOR_RANK],
        perspective: Color,
        square: usize,
        piece: Piece,
        sign: i32,
    ) {
        let relative_color = if piece.color == perspective { 0 } else { 7 };
        let piece_index = relative_color + piece_kind_index(piece.kind);
        let relative_square = canonical_square_for(perspective, square);
        if let Some(feature) = nnue::input_feature_index(piece_index * BOARD_SIZE + relative_square)
        {
            self.add_policy_row(accumulator, feature, sign);
        }
    }

    pub(crate) fn add_policy_row(
        &self,
        accumulator: &mut [f32; POLICY_ACCUMULATOR_RANK],
        row: usize,
        sign: i32,
    ) {
        let start = row * POLICY_ACCUMULATOR_RANK;
        let values = &self.policy_accumulator_features[start..start + POLICY_ACCUMULATOR_RANK];
        for (target, &value) in accumulator.iter_mut().zip(values) {
            *target += sign as f32 * value;
        }
    }

    pub(crate) fn rebuild_policy_tactical(&mut self) {
        self.policy_tactical_active = self.policy_tactical.iter().any(|&weight| weight != 0.0);
    }
}
