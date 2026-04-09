use crate::xiangqi::{BOARD_RANKS, Color, Move, PieceKind, Position};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoveNature {
    Quiet,
    Check,
    Chase(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoveRecord {
    pub resulting_hash: u64,
    pub mover: Color,
    pub nature: MoveNature,
}

#[derive(Clone, Debug, Default)]
pub struct RootMovePolicy {
    pub forbidden: Vec<Move>,
    pub forced_draw: Vec<Move>,
}

struct TargetList {
    squares: [usize; 8],
    len: usize,
}

impl TargetList {
    fn new() -> Self {
        Self {
            squares: [0; 8],
            len: 0,
        }
    }

    fn push(&mut self, sq: usize) {
        if self.len < self.squares.len() {
            self.squares[self.len] = sq;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[usize] {
        &self.squares[..self.len]
    }
}

impl RootMovePolicy {
    pub fn is_empty(&self) -> bool {
        self.forbidden.is_empty() && self.forced_draw.is_empty()
    }
}

pub fn classify_move(position: &Position, mv: Move) -> MoveNature {
    let Some(moving_piece) = position.piece_at(mv.from as usize) else {
        return MoveNature::Quiet;
    };

    let mut after = position.clone();
    after.make_move(mv);
    classify_after_move(&after, moving_piece.kind, mv.to as usize)
}

fn classify_after_move(after: &Position, moving_kind: PieceKind, attacker_sq: usize) -> MoveNature {
    if !after.has_general(after.side_to_move()) {
        return MoveNature::Check;
    }
    if after.in_check(after.side_to_move()) {
        return MoveNature::Check;
    }

    if matches!(moving_kind, PieceKind::General | PieceKind::Soldier) {
        return MoveNature::Quiet;
    }

    let defender = after.side_to_move();
    let mut chase_target = None;
    visit_attacked_targets(after, moving_kind, attacker_sq, |sq| {
        let Some(target_piece) = after.piece_at(sq) else {
            return false;
        };
        if target_piece.color != defender || target_piece.kind == PieceKind::General {
            return false;
        }
        if target_piece.kind == PieceKind::Soldier && !soldier_crossed_river(target_piece.color, sq)
        {
            return false;
        }
        if after.is_piece_protected(sq, defender) {
            return false;
        }
        if can_legally_capture_attacker(after, defender, sq, attacker_sq) {
            return false;
        }
        chase_target = Some(sq as u8);
        true
    });

    chase_target
        .map(MoveNature::Chase)
        .unwrap_or(MoveNature::Quiet)
}

pub fn make_record(position: &Position, mv: Move) -> MoveRecord {
    let mover = position.side_to_move();
    let mut after = position.clone();
    after.make_move(mv);
    let moving_kind = position
        .piece_at(mv.from as usize)
        .map(|piece| piece.kind)
        .unwrap_or(PieceKind::General);
    make_record_after_move(&after, mover, moving_kind, mv.to as usize)
}

pub fn make_record_after_move(
    after: &Position,
    mover: Color,
    moving_kind: PieceKind,
    attacker_sq: usize,
) -> MoveRecord {
    MoveRecord {
        resulting_hash: after.hash(),
        mover,
        nature: classify_after_move(after, moving_kind, attacker_sq),
    }
}

pub fn make_record_after_move_mut(
    after: &mut Position,
    mover: Color,
    moving_kind: PieceKind,
    attacker_sq: usize,
) -> MoveRecord {
    MoveRecord {
        resulting_hash: after.hash(),
        mover,
        nature: classify_after_move_mut(after, moving_kind, attacker_sq),
    }
}

pub fn derive_root_policy(position: &Position, history: &[MoveRecord]) -> RootMovePolicy {
    derive_policy_from_histories(position, history, &[])
}

fn attacked_targets(position: &Position, moving_kind: PieceKind, from: usize) -> TargetList {
    let mut targets = TargetList::new();
    visit_attacked_targets(position, moving_kind, from, |sq| {
        targets.push(sq);
        false
    });
    targets
}

fn visit_attacked_targets<F>(
    position: &Position,
    moving_kind: PieceKind,
    from: usize,
    mut visitor: F,
) where
    F: FnMut(usize) -> bool,
{
    let color = position
        .piece_at(from)
        .map(|piece| piece.color)
        .unwrap_or(Color::Red);
    match moving_kind {
        PieceKind::Rook => visit_rook_targets(position, from, &mut visitor),
        PieceKind::Cannon => visit_cannon_targets(position, from, &mut visitor),
        PieceKind::Horse => visit_horse_targets(position, from, &mut visitor),
        PieceKind::Advisor => visit_advisor_targets(color, from, &mut visitor),
        PieceKind::Elephant => visit_elephant_targets(position, color, from, &mut visitor),
        _ => {}
    };
}

fn can_legally_capture_attacker(
    position: &Position,
    mover: Color,
    from: usize,
    target: usize,
) -> bool {
    if !position.piece_attacks_square_from(from, target) {
        return false;
    }

    let mut work = position.clone();
    let mv = Move::new(from, target);
    let undo = work.make_move(mv);
    let legal = !work.in_check(mover);
    work.unmake_move(mv, undo);
    legal
}

fn classify_after_move_mut(
    after: &mut Position,
    moving_kind: PieceKind,
    attacker_sq: usize,
) -> MoveNature {
    if !after.has_general(after.side_to_move()) {
        return MoveNature::Check;
    }
    if after.in_check(after.side_to_move()) {
        return MoveNature::Check;
    }

    if matches!(moving_kind, PieceKind::General | PieceKind::Soldier) {
        return MoveNature::Quiet;
    }

    let defender = after.side_to_move();
    let targets = attacked_targets(after, moving_kind, attacker_sq);
    let mut chase_target = None;
    for &sq in targets.as_slice() {
        let Some(target_piece) = after.piece_at(sq) else {
            continue;
        };
        if target_piece.color != defender || target_piece.kind == PieceKind::General {
            continue;
        }
        if target_piece.kind == PieceKind::Soldier && !soldier_crossed_river(target_piece.color, sq)
        {
            continue;
        }
        if after.is_piece_protected(sq, defender) {
            continue;
        }
        if can_legally_capture_attacker_in_place(after, defender, sq, attacker_sq) {
            continue;
        }
        chase_target = Some(sq as u8);
        break;
    }

    chase_target
        .map(MoveNature::Chase)
        .unwrap_or(MoveNature::Quiet)
}

fn can_legally_capture_attacker_in_place(
    position: &mut Position,
    mover: Color,
    from: usize,
    target: usize,
) -> bool {
    if !position.piece_attacks_square_from(from, target) {
        return false;
    }

    let mv = Move::new(from, target);
    let undo = position.make_move(mv);
    let legal = !position.in_check(mover);
    position.unmake_move(mv, undo);
    legal
}

fn visit_rook_targets<F>(position: &Position, from: usize, visitor: &mut F)
where
    F: FnMut(usize) -> bool,
{
    visit_slider_targets(position, from, false, visitor)
}

fn visit_cannon_targets<F>(position: &Position, from: usize, visitor: &mut F)
where
    F: FnMut(usize) -> bool,
{
    visit_slider_targets(position, from, true, visitor)
}

fn visit_slider_targets<F>(position: &Position, from: usize, cannon: bool, visitor: &mut F)
where
    F: FnMut(usize) -> bool,
{
    let file = (from % 9) as i32;
    let rank = (from / 9) as i32;

    for (df, dr) in [(1_i32, 0_i32), (-1, 0), (0, 1), (0, -1)] {
        let mut nf = file + df;
        let mut nr = rank + dr;
        let mut screens = 0usize;
        while inside_board(nf, nr) {
            let sq = index(nf as usize, nr as usize);
            if position.piece_at(sq).is_some() {
                if !cannon {
                    if visitor(sq) {
                        return;
                    }
                    break;
                }
                screens += 1;
                if screens == 2 {
                    if visitor(sq) {
                        return;
                    }
                    break;
                }
            }
            nf += df;
            nr += dr;
        }
    }
}

fn visit_horse_targets<F>(position: &Position, from: usize, visitor: &mut F)
where
    F: FnMut(usize) -> bool,
{
    let file = (from % 9) as i32;
    let rank = (from / 9) as i32;

    for ((leg_df, leg_dr), (move_df, move_dr)) in [
        ((0, -1), (-1, -2)),
        ((0, -1), (1, -2)),
        ((1, 0), (2, -1)),
        ((1, 0), (2, 1)),
        ((0, 1), (1, 2)),
        ((0, 1), (-1, 2)),
        ((-1, 0), (-2, 1)),
        ((-1, 0), (-2, -1)),
    ] {
        let leg_f = file + leg_df;
        let leg_r = rank + leg_dr;
        let target_f = file + move_df;
        let target_r = rank + move_dr;
        if inside_board(target_f, target_r)
            && position
                .piece_at(index(leg_f as usize, leg_r as usize))
                .is_none()
            && visitor(index(target_f as usize, target_r as usize))
        {
            return;
        }
    }
}

fn visit_advisor_targets<F>(color: Color, from: usize, visitor: &mut F)
where
    F: FnMut(usize) -> bool,
{
    let file = (from % 9) as i32;
    let rank = (from / 9) as i32;

    for (df, dr) in [(1_i32, 1_i32), (1, -1), (-1, 1), (-1, -1)] {
        let nf = file + df;
        let nr = rank + dr;
        if inside_board(nf, nr)
            && inside_palace(color, nf as usize, nr as usize)
            && visitor(index(nf as usize, nr as usize))
        {
            return;
        }
    }
}

fn visit_elephant_targets<F>(position: &Position, color: Color, from: usize, visitor: &mut F)
where
    F: FnMut(usize) -> bool,
{
    let file = (from % 9) as i32;
    let rank = (from / 9) as i32;

    for (df, dr) in [(2_i32, 2_i32), (2, -2), (-2, 2), (-2, -2)] {
        let nf = file + df;
        let nr = rank + dr;
        let eye_f = file + df / 2;
        let eye_r = rank + dr / 2;
        if inside_board(nf, nr)
            && elephant_stays_home(color, nr as usize)
            && position
                .piece_at(index(eye_f as usize, eye_r as usize))
                .is_none()
            && visitor(index(nf as usize, nr as usize))
        {
            return;
        }
    }
}

fn inside_board(file: i32, rank: i32) -> bool {
    (0..9).contains(&file) && (0..10).contains(&rank)
}

fn index(file: usize, rank: usize) -> usize {
    rank * 9 + file
}

fn inside_palace(color: Color, file: usize, rank: usize) -> bool {
    (3..=5).contains(&file)
        && match color {
            Color::Black => rank <= 2,
            Color::Red => rank >= 7,
        }
}

fn elephant_stays_home(color: Color, rank: usize) -> bool {
    match color {
        Color::Black => rank <= 4,
        Color::Red => rank >= BOARD_RANKS - 5,
    }
}

pub fn derive_policy_from_histories(
    position: &Position,
    history_prefix: &[MoveRecord],
    history_suffix: &[MoveRecord],
) -> RootMovePolicy {
    let mut policy = RootMovePolicy::default();

    for mv in position.legal_moves() {
        let record = make_record(position, mv);
        let (count, previous, latest) =
            matching_repetition_tail(record, history_prefix, history_suffix);
        if count < 2 {
            continue;
        }

        let recent = [previous.unwrap(), latest.unwrap(), record];
        if recent.iter().all(|entry| entry.nature == MoveNature::Check) {
            policy.forbidden.push(mv);
            continue;
        }

        let chase_target = match recent[0].nature {
            MoveNature::Chase(target) => Some(target),
            _ => None,
        };
        if let Some(target) = chase_target {
            if recent[1].nature == MoveNature::Chase(target)
                && recent[2].nature == MoveNature::Chase(target)
            {
                policy.forbidden.push(mv);
                continue;
            }
        }

        policy.forced_draw.push(mv);
    }

    policy
}

fn matching_repetition_tail(
    target: MoveRecord,
    history_prefix: &[MoveRecord],
    history_suffix: &[MoveRecord],
) -> (usize, Option<MoveRecord>, Option<MoveRecord>) {
    let mut count = 0usize;
    let mut previous = None;
    let mut latest = None;

    for entry in history_prefix.iter().chain(history_suffix.iter()) {
        if entry.resulting_hash == target.resulting_hash && entry.mover == target.mover {
            count += 1;
            previous = latest;
            latest = Some(*entry);
        }
    }

    (count, previous, latest)
}

fn soldier_crossed_river(color: Color, sq: usize) -> bool {
    let rank = sq / 9;
    match color {
        Color::Black => rank >= 5,
        Color::Red => rank <= BOARD_RANKS - 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xiangqi::STARTPOS_FEN;

    #[test]
    fn classifies_checking_move() {
        let position = Position::from_fen("4k4/9/9/9/9/9/4R4/9/9/4K4 w").unwrap();
        let mv = Move::from_uci("e3e9").unwrap();
        assert_eq!(classify_move(&position, mv), MoveNature::Check);
    }

    #[test]
    fn detects_repeated_check_as_forbidden_root_move() {
        let mut position = Position::startpos();
        let mut records = Vec::new();
        for mv_text in [
            "h2e2", "h7e7", "e2h2", "e7h7", "h2e2", "h7e7", "e2h2", "e7h7",
        ] {
            let mv = position.parse_uci_move(mv_text).unwrap();
            records.push(make_record(&position, mv));
            position.make_move(mv);
        }

        let policy = derive_root_policy(&position, &records);
        assert_eq!(position.to_fen(), STARTPOS_FEN);
        assert!(
            policy
                .forced_draw
                .contains(&Move::from_uci("h2e2").unwrap())
                || policy.forbidden.contains(&Move::from_uci("h2e2").unwrap())
        );
    }

    #[test]
    fn chase_detection_requires_actual_capture_geometry() {
        let position = Position::from_fen("4k4/9/2b6/9/4p4/2R6/9/9/9/4K4 w").unwrap();
        let mv = Move::from_uci("c4c5").unwrap();
        assert!(matches!(classify_move(&position, mv), MoveNature::Chase(_)));
    }
}
