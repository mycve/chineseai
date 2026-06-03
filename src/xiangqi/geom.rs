use super::{BOARD_FILES, BOARD_RANKS, Color, HORSE_STEPS};

pub fn square_name(sq: usize) -> String {
    let file = (b'a' + file_of(sq) as u8) as char;
    let rank = (BOARD_RANKS - 1 - rank_of(sq)).to_string();
    format!("{file}{rank}")
}

pub fn parse_square(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.len() != 2 {
        return None;
    }

    let file = bytes[0].to_ascii_lowercase();
    let rank = bytes[1];
    if !(b'a'..=b'i').contains(&file) || !rank.is_ascii_digit() {
        return None;
    }

    let file_idx = (file - b'a') as usize;
    let rank_from_bottom = (rank - b'0') as usize;
    let rank_idx = BOARD_RANKS - 1 - rank_from_bottom;
    Some(index(file_idx, rank_idx))
}

#[inline(always)]
pub(super) fn index(file: usize, rank: usize) -> usize {
    rank * BOARD_FILES + file
}

#[inline(always)]
pub(super) fn file_of(sq: usize) -> usize {
    sq % BOARD_FILES
}

#[inline(always)]
pub(super) fn rank_of(sq: usize) -> usize {
    sq / BOARD_FILES
}

#[inline(always)]
pub(super) fn inside_board(file: i32, rank: i32) -> bool {
    (0..BOARD_FILES as i32).contains(&file) && (0..BOARD_RANKS as i32).contains(&rank)
}

pub(super) fn inside_palace(color: Color, file: usize, rank: usize) -> bool {
    let file_ok = (3..=5).contains(&file);
    let rank_ok = match color {
        Color::Black => (0..=2).contains(&rank),
        Color::Red => (7..=9).contains(&rank),
    };
    file_ok && rank_ok
}

pub(super) fn elephant_stays_home(color: Color, rank: usize) -> bool {
    match color {
        Color::Black => rank <= 4,
        Color::Red => rank >= 5,
    }
}

pub(super) fn soldier_crossed_river(color: Color, rank: usize) -> bool {
    match color {
        Color::Black => rank >= 5,
        Color::Red => rank <= 4,
    }
}

#[cfg(test)]
#[inline(always)]
pub(super) fn same_rank_or_file(a: usize, b: usize) -> bool {
    file_of(a) == file_of(b) || rank_of(a) == rank_of(b)
}

pub(super) fn line_between_squares(a: usize, b: usize) -> Vec<usize> {
    let mut squares = Vec::new();
    if file_of(a) == file_of(b) {
        let file = file_of(a);
        let start = rank_of(a).min(rank_of(b)) + 1;
        let end = rank_of(a).max(rank_of(b));
        for rank in start..end {
            squares.push(index(file, rank));
        }
    } else if rank_of(a) == rank_of(b) {
        let rank = rank_of(a);
        let start = file_of(a).min(file_of(b)) + 1;
        let end = file_of(a).max(file_of(b));
        for file in start..end {
            squares.push(index(file, rank));
        }
    }
    squares
}

pub(super) fn horse_leg_square(from: usize, target: usize) -> Option<usize> {
    let file = file_of(from) as i32;
    let rank = rank_of(from) as i32;
    let tf = file_of(target) as i32;
    let tr = rank_of(target) as i32;
    let df = tf - file;
    let dr = tr - rank;

    for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
        if df == move_df && dr == move_dr {
            let leg_file = file + leg_df;
            let leg_rank = rank + leg_dr;
            if inside_board(leg_file, leg_rank) {
                return Some(index(leg_file as usize, leg_rank as usize));
            }
        }
    }
    None
}
