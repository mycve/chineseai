use super::{
    Color, Move, Piece, Position, SIDE_TO_MOVE_KEY, color_hash_index, file_of, index, rank_of,
    zobrist_piece_key,
};

impl Position {
    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    #[inline(always)]
    pub fn piece_at(&self, sq: usize) -> Option<Piece> {
        self.board[sq]
    }

    #[inline(always)]
    pub fn has_general(&self, color: Color) -> bool {
        self.general_squares[color_hash_index(color)].is_some()
    }

    #[inline(always)]
    pub fn general_square(&self, color: Color) -> Option<usize> {
        self.general_squares[color_hash_index(color)]
    }

    #[cfg(test)]
    #[inline(always)]
    pub(super) fn has_dynamic_material(&self, color: Color) -> bool {
        self.dynamic_material_counts[color_hash_index(color)] > 0
    }

    #[inline(always)]
    pub fn is_capture(&self, mv: Move) -> bool {
        self.board[mv.to as usize].is_some()
    }

    #[inline(always)]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub(super) fn hash_after_move(&self, mv: Move) -> u64 {
        let from = mv.from as usize;
        let to = mv.to as usize;
        let moving = self.board[from].expect("move from occupied square");
        let mut hash = self.hash ^ zobrist_piece_key(from, moving);
        if let Some(captured) = self.board[to] {
            hash ^= zobrist_piece_key(to, captured);
        }
        hash ^ zobrist_piece_key(to, moving) ^ SIDE_TO_MOVE_KEY
    }

    #[inline(always)]
    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    #[inline(always)]
    pub fn rule60_max_ply(&self) -> Option<u16> {
        self.rule60_max_ply
    }

    pub fn repetition_draw_enabled(&self) -> bool {
        self.repetition_draw_enabled
    }

    /// 仅控制普通重复局面判和，不关闭长将、长捉判负或其他终局规则。
    pub fn set_repetition_draw_enabled(&mut self, enabled: bool) {
        self.repetition_draw_enabled = enabled;
    }

    /// `sq` 是否被 `color` 方攻击。
    ///
    /// 名字里的"保护"只是习惯叫法：语义就是"该格被该方攻击"，不检查 `sq` 上站的是谁
    /// （是对方子、己方子还是空格，结果一样）。因为"没有棋子攻击自己所在的格子"对
    /// 跃子和滑子都成立，所以它和 `is_square_attacked(sq, color)` 完全等价。
    #[inline(always)]
    pub fn is_piece_protected(&self, sq: usize, color: Color) -> bool {
        self.is_square_attacked(sq, color)
    }

    pub(super) fn find_general(&self, color: Color) -> Option<usize> {
        self.general_squares[color_hash_index(color)]
    }

    pub(super) fn generals_face(&self) -> bool {
        let red = self.find_general(Color::Red).unwrap();
        let black = self.find_general(Color::Black).unwrap();
        file_of(red) == file_of(black) && self.clear_file_between(red, black)
    }

    pub(super) fn clear_file_between(&self, a: usize, b: usize) -> bool {
        if file_of(a) != file_of(b) {
            return false;
        }

        let file = file_of(a);
        let start = rank_of(a).min(rank_of(b)) + 1;
        let end = rank_of(a).max(rank_of(b));

        for rank in start..end {
            if self.board[index(file, rank)].is_some() {
                return false;
            }
        }
        true
    }
}
