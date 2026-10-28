//! 局面上下文的唯一布局；训练与搜索共用，候选走法标记留在策略头。
use super::*;
use crate::xiangqi::BOARD_FILES;

pub(crate) fn position_context_features(
    rules: &[f32; RULE_CONTEXT_SIZE],
    checks: &[f32; CHECK_CONTEXT_SIZE],
    history: &[f32; HISTORY_CONTEXT_SIZE],
) -> [f32; POSITION_CONTEXT_SIZE] {
    let mut context = [0.0; POSITION_CONTEXT_SIZE];
    context[..CHECK_CONTEXT_OFFSET].copy_from_slice(rules);
    context[CHECK_CONTEXT_OFFSET..HISTORY_CONTEXT_OFFSET].copy_from_slice(checks);
    context[HISTORY_CONTEXT_OFFSET..].copy_from_slice(history);
    context
}

/// 九宫的 9 个格子（按颜色）。
fn palace_mask(color: Color) -> u128 {
    let ranks: [usize; 3] = match color {
        Color::Red => [7, 8, 9],
        Color::Black => [0, 1, 2],
    };
    let mut mask = 0u128;
    for rank in ranks {
        for file in 3..=5 {
            mask |= 1u128 << (rank * BOARD_FILES + file);
        }
    }
    mask
}

/// 同线且中间全空（用于飞将判断）。
fn file_clear_between(position: &Position, a: usize, b: usize) -> bool {
    let file = a % BOARD_FILES;
    if b % BOARD_FILES != file {
        return false;
    }
    let (ra, rb) = (a / BOARD_FILES, b / BOARD_FILES);
    let (start, end) = if ra < rb { (ra + 1, rb) } else { (rb + 1, ra) };
    (start..end).all(|rank| position.piece_at(rank * BOARD_FILES + file).is_none())
}

/// `color` 的将还有几个**安全逃格**：宫内相邻、不被己方子挡住、不被 `attacker_mask` 覆盖、
/// 且走进去不会与对方将照面。
///
/// 0 就是"杀网已经成形"。这里用对方攻击位板做判定，正好复用策略头已经算好的那张位板。
fn king_safe_escapes(position: &Position, color: Color, attacker_mask: u128) -> usize {
    let Some(king) = position.general_square(color) else {
        return 0;
    };
    let enemy_king = position.general_square(color.opposite());
    let file = king % BOARD_FILES;
    let rank = king / BOARD_FILES;
    let mut safe = 0usize;
    for (df, dr) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
        let nf = file as i32 + df;
        let nr = rank as i32 + dr;
        if !(3..=5).contains(&nf) {
            continue;
        }
        let rank_ok = match color {
            Color::Red => (7..=9).contains(&nr),
            Color::Black => (0..=2).contains(&nr),
        };
        if !rank_ok {
            continue;
        }
        let square = nr as usize * BOARD_FILES + nf as usize;
        if position
            .piece_at(square)
            .is_some_and(|piece| piece.color == color)
        {
            continue;
        }
        if attacker_mask & (1u128 << square) != 0 {
            continue;
        }
        if enemy_king.is_some_and(|enemy| {
            enemy % BOARD_FILES == square % BOARD_FILES
                && file_clear_between(position, enemy, square)
        }) {
            continue;
        }
        safe += 1;
    }
    safe
}

/// 引擎**已经算过、却一直没喂给模型**的 8 个标量。全部来自调用方已经算好的量：
/// `gives_check`（`fill_policy_gives_checks` 本来就无条件算）、`masks`（策略头本来就要的
/// 全盘攻击位板）、`moves.len()`（调用方手里就有）。

pub(crate) fn check_context_features(
    position: &Position,
    moves: &[Move],
    gives_check: &[f32],
    masks: [u128; 2],
) -> [f32; CHECK_CONTEXT_SIZE] {
    let side = position.side_to_move();
    let enemy = side.opposite();
    let own_attacks = masks[color_index(side)];
    let enemy_attacks = masks[color_index(enemy)];
    let checks = gives_check.iter().filter(|&&flag| flag != 0.0).count();
    // 复用当前局面的敌方攻击位棋盘；缺少将位时保持原来的 false。
    let in_check = position
        .general_square(side)
        .is_some_and(|square| enemy_attacks & (1u128 << square) != 0);
    let enemy_escapes = king_safe_escapes(position, enemy, own_attacks);
    let our_escapes = king_safe_escapes(position, side, enemy_attacks);
    [
        f32::from(in_check),
        (checks as f32 / 4.0).min(1.0),
        (enemy_escapes as f32 / 4.0).min(1.0),
        (our_escapes as f32 / 4.0).min(1.0),
        ((own_attacks & palace_mask(enemy)).count_ones() as f32 / 9.0).min(1.0),
        ((enemy_attacks & palace_mask(side)).count_ones() as f32 / 9.0).min(1.0),
        (moves.len() as f32 / 64.0).min(1.0),
        // 线性层拼不出这个交互："我有将军着法、而对方的将几乎没有安全逃格"。
        f32::from(checks > 0 && enemy_escapes <= 1),
    ]
}

impl AzNnue {
    pub(crate) fn fill_policy_gives_checks(
        &self,
        position: &Position,
        moves: &[Move],
        output: &mut Vec<f32>,
    ) {
        crate::scope_profile!("az.eval.policy.gives_check");
        output.resize(moves.len(), 0.0);
        for (flag, &mv) in output.iter_mut().zip(moves) {
            *flag = f32::from(position.gives_check_after_move_fast(mv));
        }
    }

    /// 在主干之前把"每走法将军 flag + 双方攻击位板"算好一次，供标量块与策略头共用。
    ///
    /// 这两样东西策略头本来就要算（位板受 `policy_tactical_active` 门控、flag 无条件），
    /// 而标量块需要在主干之前用它们，所以只能提前算：总工作量不变，只是把顺序挪前。
    pub(crate) fn fill_policy_inputs(
        &self,
        position: &Position,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
        with_masks: bool,
    ) {
        self.fill_policy_gives_checks(position, moves, &mut scratch.policy_gives_check);
        if with_masks {
            scratch.attack_masks = position.attacked_squares_masks();
        }
        scratch.policy_inputs_ready = true;
    }

    pub(crate) fn rebuild_context(&mut self) {
        self.context_active = self.context_hidden.iter().any(|&weight| weight != 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_changes_policy_value_and_moves_left_with_one_shared_context() {
        let mut model = AzNnue::random(32, 20261028);
        let past = (HISTORY_CONTEXT_OFFSET + HISTORY_CONTEXT_SIZE / 2) * 2 * model.hidden_size;
        for (i, w) in model.context_hidden[past..].iter_mut().enumerate() {
            *w = (i % 29) as f32 * 0.004 - 0.056;
        }
        model.rebuild_context();
        model.value_head_bias.fill(1.0);
        for (i, w) in model.value_head_output.iter_mut().enumerate() {
            *w = (i % 17) as f32 * 0.005 - 0.04;
        }
        for (i, w) in model.policy_move_context.iter_mut().enumerate() {
            *w = (i % 31) as f32 * 0.004 - 0.06;
        }
        for (i, w) in model.moves_left_output.iter_mut().enumerate() {
            *w = (i % 13) as f32 * 0.004 - 0.024;
        }
        model.rebuild_moves_left();
        let mut position = Position::startpos();
        let mut history = position.initial_rule_history();
        let mut full = AzEvalScratch::new(model.arch);
        let mut incremental = AzEvalScratch::new(model.arch);
        let mut without = AzEvalScratch::new(model.arch);
        let mut checked = 0;
        for _ in 0..6 {
            let moves = position.legal_moves();
            let mv = moves[0];
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
            let moves = position.legal_moves();
            let features = history_features(&position, &history);
            let empty = history_features(&position, &[]);
            assert_eq!(features[..48], empty[..48]);
            assert!(features[48..].iter().any(|x| *x != 0.0));
            let with = model.evaluate_with_scratch_output_with_repetition_and_history_features(
                &position,
                &moves,
                &[],
                &[0.0; RULE_CONTEXT_SIZE],
                &features,
                &mut full,
            );
            let no_history = model
                .evaluate_with_scratch_output_with_repetition_and_history_features(
                    &position,
                    &moves,
                    &[],
                    &[0.0; RULE_CONTEXT_SIZE],
                    &empty,
                    &mut without,
                );
            assert!(
                with.value_wdl
                    .iter()
                    .zip(no_history.value_wdl)
                    .any(|(a, b)| (a - b).abs() > 1e-7)
            );
            assert!((with.moves_left - no_history.moves_left).abs() > 1e-6);
            let first_shift = full.logits[0] - without.logits[0];
            assert!(
                full.logits
                    .iter()
                    .zip(&without.logits)
                    .any(|(a, b)| (a - b - first_shift).abs() > 1e-7),
                "历史只能改变公共偏置，无法改变走法偏好"
            );
            let accumulator = AzEvalAccumulator::new(&model, &position);
            let policy = model.policy_accumulator(&position, position.side_to_move());
            let result = model.evaluate_incremental_with_scratch_output_with_history(
                &position,
                &accumulator.into_hidden_sum(),
                &policy,
                &moves,
                &[],
                &[0.0; RULE_CONTEXT_SIZE],
                &history,
                &mut incremental,
            );
            assert_eq!(with.value_wdl, result.value_wdl);
            assert_eq!(with.moves_left, result.moves_left);
            assert_eq!(full.logits, incremental.logits);
            checked += 1;
        }
        assert_eq!(checked, 6);
        let path = std::env::temp_dir().join(format!(
            "chineseai_context_{}.safetensors",
            std::process::id()
        ));
        model.save(&path).unwrap();
        let loaded = AzNnue::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(loaded.context_hidden, model.context_hidden);
        let moves = position.legal_moves();
        let output = loaded.evaluate_with_scratch_output_with_repetition_and_history(
            &position,
            &moves,
            &[],
            &[0.0; RULE_CONTEXT_SIZE],
            &history,
            &mut incremental,
        );
        let original = model.evaluate_with_scratch_output_with_repetition_and_history(
            &position,
            &moves,
            &[],
            &[0.0; RULE_CONTEXT_SIZE],
            &history,
            &mut full,
        );
        assert_eq!(output.value_wdl, original.value_wdl);
        assert_eq!(full.logits, incremental.logits);
    }
}
