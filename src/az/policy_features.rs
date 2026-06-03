use crate::xiangqi::{BOARD_SIZE, Color, Move, Piece, PieceKind, Position, piece_base_value};

use super::POLICY_DYNAMIC_FEATURE_SIZE;

pub(crate) fn policy_dynamic_features_cached(
    position: &Position,
    mv: Move,
    cache: &PolicyFeaturePositionCache,
) -> [f32; POLICY_DYNAMIC_FEATURE_SIZE] {
    crate::scope_profile!("az.policy_features.total");
    let mut features = PolicyDynamicFeatureBuilder::new();
    let Some(context) = MoveFeatureContext::new(position, mv, cache) else {
        return features.finish();
    };

    add_piece_identity_features(&context, &mut features);
    add_material_features(&context, &mut features);
    add_safety_features(&context, &mut features);
    add_after_move_features(&context, &mut features);
    features.add_bias();
    features.finish()
}

pub(crate) struct PolicyFeaturePositionCache {
    side: Color,
    them: Color,
    before_in_check_side: bool,
    attacked_by_side: [bool; BOARD_SIZE],
    attacked_by_them: [bool; BOARD_SIZE],
}

impl PolicyFeaturePositionCache {
    pub(crate) fn new_for_moves(position: &Position, moves: &[Move]) -> Self {
        let side = position.side_to_move();
        let them = side.opposite();
        let before_in_check_side = position.in_check(side);
        let mut attacked_by_side = [false; BOARD_SIZE];
        let mut attacked_by_them = [false; BOARD_SIZE];
        let mut side_done = [false; BOARD_SIZE];
        let mut them_done = [false; BOARD_SIZE];
        for mv in moves {
            let from = mv.from as usize;
            let to = mv.to as usize;
            if !them_done[from] {
                attacked_by_them[from] = position.is_square_attacked(from, them);
                them_done[from] = true;
            }
            if !them_done[to] {
                attacked_by_them[to] = position.is_square_attacked(to, them);
                them_done[to] = true;
            }
            if !side_done[to] {
                attacked_by_side[to] = position.is_square_attacked(to, side);
                side_done[to] = true;
            }
        }
        Self {
            side,
            them,
            before_in_check_side,
            attacked_by_side,
            attacked_by_them,
        }
    }
}

struct MoveFeatureContext<'a> {
    after: Position,
    cache: &'a PolicyFeaturePositionCache,
    mv: Move,
    moved: Piece,
    captured: Option<Piece>,
    moved_value: f32,
    captured_value: f32,
}

impl<'a> MoveFeatureContext<'a> {
    fn new(before: &'a Position, mv: Move, cache: &'a PolicyFeaturePositionCache) -> Option<Self> {
        let moved = before.piece_at(mv.from as usize)?;
        let captured = before.piece_at(mv.to as usize);
        let moved_value = piece_base_value(moved.kind).max(1) as f32;
        let captured_value = captured
            .map(|piece| piece_base_value(piece.kind).max(0) as f32)
            .unwrap_or(0.0);
        let mut after = before.clone();
        after.make_move(mv);
        Some(Self {
            after,
            cache,
            mv,
            moved,
            captured,
            moved_value,
            captured_value,
        })
    }
}

struct PolicyDynamicFeatureBuilder {
    values: [f32; POLICY_DYNAMIC_FEATURE_SIZE],
    next: usize,
}

impl PolicyDynamicFeatureBuilder {
    fn new() -> Self {
        Self {
            values: [0.0; POLICY_DYNAMIC_FEATURE_SIZE],
            next: 0,
        }
    }

    fn push(&mut self, value: f32) {
        debug_assert!(
            self.next < POLICY_DYNAMIC_FEATURE_SIZE,
            "too many policy dynamic features"
        );
        if self.next < POLICY_DYNAMIC_FEATURE_SIZE {
            self.values[self.next] = value;
            self.next += 1;
        }
    }

    fn push_bool(&mut self, value: bool) {
        self.push(value as u8 as f32);
    }

    fn push_piece_kind_one_hot(&mut self, kind: Option<PieceKind>) {
        for expected in PIECE_KINDS {
            self.push_bool(kind == Some(expected));
        }
    }

    fn add_bias(&mut self) {
        while self.next + 1 < POLICY_DYNAMIC_FEATURE_SIZE {
            self.push(0.0);
        }
        self.push(1.0);
    }

    fn finish(self) -> [f32; POLICY_DYNAMIC_FEATURE_SIZE] {
        self.values
    }
}

const PIECE_KINDS: [PieceKind; 7] = [
    PieceKind::General,
    PieceKind::Advisor,
    PieceKind::Elephant,
    PieceKind::Horse,
    PieceKind::Rook,
    PieceKind::Cannon,
    PieceKind::Soldier,
];

fn add_piece_identity_features(
    context: &MoveFeatureContext<'_>,
    features: &mut PolicyDynamicFeatureBuilder,
) {
    crate::scope_profile!("az.policy_features.identity");
    features.push_piece_kind_one_hot(Some(context.moved.kind));
    features.push_bool(context.moved.color == context.cache.side);
    features.push_piece_kind_one_hot(context.captured.map(|piece| piece.kind));
    features.push_bool(context.captured.is_some());
}

fn add_material_features(
    context: &MoveFeatureContext<'_>,
    features: &mut PolicyDynamicFeatureBuilder,
) {
    crate::scope_profile!("az.policy_features.material");
    let material_delta = context.captured_value - context.moved_value;
    features.push((context.captured_value / 900.0).clamp(0.0, 1.0));
    features.push((context.moved_value / 900.0).clamp(0.0, 1.0));
    features.push((material_delta / 900.0).clamp(-1.0, 1.0));
}

fn add_safety_features(
    context: &MoveFeatureContext<'_>,
    features: &mut PolicyDynamicFeatureBuilder,
) {
    crate::scope_profile!("az.policy_features.safety");
    let from = context.mv.from as usize;
    let to = context.mv.to as usize;
    features.push_bool(
        context.moved_value > context.captured_value && context.cache.attacked_by_them[to],
    );
    features.push_bool(context.cache.attacked_by_them[from]);
    features.push_bool(context.cache.attacked_by_them[to]);
    features.push_bool(context.cache.attacked_by_side[to]);
    features.push_bool(context.cache.before_in_check_side);
}

fn add_after_move_features(
    context: &MoveFeatureContext<'_>,
    features: &mut PolicyDynamicFeatureBuilder,
) {
    crate::scope_profile!("az.policy_features.after_move");
    let to = context.mv.to as usize;
    let gives_check = context.after.in_check(context.cache.them);
    let destination_attacked = context.after.is_square_attacked(to, context.cache.them);
    let destination_protected = context.after.is_square_attacked(to, context.cache.side);

    features.push_bool(gives_check);
    features.push_bool(!context.after.in_check(context.cache.side));
    features.push_bool(destination_attacked);
    features.push_bool(destination_protected);
    features.push_bool(gives_check && context.moved_value > context.captured_value);
    features.push_bool(gives_check || context.captured.is_some());
    features.push_bool(context.cache.attacked_by_them[to] && destination_protected);
}
