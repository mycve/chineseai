//! 棋盘关系、静态战术模式及准确走法后果。训练和推理共用。
use smallvec::SmallVec;
use std::sync::OnceLock;

use super::nnue::{canonical_square, piece_absolute_feature_index};
use crate::xiangqi::{
    BOARD_SIZE, Color, Move, Piece, PieceKind, Position, color_index, piece_kind_index,
};

pub const RELATION_RANK: usize = 24;
pub const RELATION_CORE_ROWS: usize = 63_308;
pub const RELATION_ROWS: usize = RELATION_CORE_ROWS + 2 * 14 * 90;
pub const MOTIF_RANK: usize = 32;
pub const MOTIF_FAMILIES: usize = 12;
pub const MOTIF_CORE_ROWS: usize = 43_008;
pub const MOTIF_GEOMETRY_ROWS: usize = 12_288;
pub const MOTIF_ROWS: usize = MOTIF_CORE_ROWS + MOTIF_GEOMETRY_ROWS + 2_160;
pub const MOVE_LOCAL_SIZE: usize = 12;
pub const POLICY_JOINT_ROWS: usize = 7 * 8 * 4 * 4 * 2;
pub const POLICY_EVENT_ROWS: usize = MOTIF_FAMILIES * 7 * 8 * 2;
pub(crate) fn repetition_flag(repeats: bool, history_known: bool) -> u8 {
    if repeats {
        1
    } else if history_known {
        0
    } else {
        2
    }
}
pub const FEATURE_SCHEMA: &[f32] = &[1., 96., 16., 24., 32., 18., 14., 12., 63308., 57456.];
pub(crate) fn feature_schema() -> Vec<f32> {
    // 固定ASCII描述避免操作系统换行方式改变检查点身份。
    let descriptor = concat!(
        "xiangqi;tactical-v1;rotate180;piece=KAENRCP;relative-colors;roles-sort=kind,canonical-square;",
        "R=14x2x7x323+2x1260;Mcore=12x2x7x8x8x4;Mgeo=12x2x128x4;Mpos=12x2x90;",
        "M=rook-gate,cannon-screen,leaper-gate,fork,converge,guard-threat,shared-guard,exchange,battery,king-gate,check-seal,discovery;",
        "distance8=0,1,2,3,4,5-6,7-9,10+;distance4=0,1,2-3,4+;amount=1,2,3-4,5+;",
        "variant=role-relative-color;royal=post-move-origins;policy=post-move-no-self-support,content-aware-edge-delta;repeat=0,1,unknown2;",
        "rules=enable,count120,remaining120,matches3,cycle32,checks4,chases4,known9;history=complete-since-reset;"
    );
    let crc = crc32fast::hash(descriptor.as_bytes());
    let mut schema = FEATURE_SCHEMA.to_vec();
    schema.extend([(crc & 0xffff) as f32, (crc >> 16) as f32]);
    schema
}

#[derive(Clone, Copy)]
pub(crate) struct Board {
    pub cells: [Option<Piece>; BOARD_SIZE],
    pub occupied: u128,
    pub colors: [u128; 2],
    pub kings: [usize; 2],
}

pub(crate) fn bits(mut mask: u128) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let square = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        Some(square)
    })
}

fn sq(file: i32, rank: i32) -> Option<usize> {
    ((0..9).contains(&file) && (0..10).contains(&rank)).then_some((rank * 9 + file) as usize)
}

fn palace(side: Color, square: usize) -> bool {
    (3..=5).contains(&(square % 9))
        && if side == Color::Red {
            square / 9 >= 7
        } else {
            square / 9 <= 2
        }
}

const HORSE: [(i32, i32, i32, i32); 8] = [
    (-1, -2, 0, -1),
    (1, -2, 0, -1),
    (-1, 2, 0, 1),
    (1, 2, 0, 1),
    (-2, -1, -1, 0),
    (-2, 1, -1, 0),
    (2, -1, 1, 0),
    (2, 1, 1, 0),
];
const ELEPHANT: [(i32, i32, i32, i32); 4] = [
    (-2, -2, -1, -1),
    (2, -2, 1, -1),
    (-2, 2, -1, 1),
    (2, 2, 1, 1),
];
const DIRS: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

struct Geometry {
    rays: [[u128; 4]; 90],
    steps: Vec<Vec<(usize, Option<usize>, usize)>>,
    dependencies: [[u128; 90]; 14],
    unblocked: [[u128; 90]; 14],
    gates: Vec<SmallVec<[(u8, u128); 4]>>,
}

fn geometry() -> &'static Geometry {
    static DATA: OnceLock<Geometry> = OnceLock::new();
    DATA.get_or_init(|| {
        let mut g = Geometry {
            rays: [[0; 4]; 90],
            steps: vec![vec![]; 14 * 90],
            dependencies: [[0; 90]; 14],
            unblocked: [[0; 90]; 14],
            gates: vec![SmallVec::new(); 14 * 90],
        };
        for source in 0..90 {
            let (f, r) = ((source % 9) as i32, (source / 9) as i32);
            for (d, (df, dr)) in DIRS.into_iter().enumerate() {
                let (mut nf, mut nr) = (f + df, r + dr);
                while let Some(target) = sq(nf, nr) {
                    g.rays[source][d] |= 1u128 << target;
                    nf += df;
                    nr += dr;
                }
            }
            for class in 0..14 {
                let side = if class < 7 { Color::Red } else { Color::Black };
                let kind = class % 7;
                let mut steps = vec![];
                if kind == 4 || kind == 5 || kind == 0 {
                    g.dependencies[class][source] =
                        g.rays[source].iter().copied().fold(0, |a, b| a | b);
                }
                let candidates: Vec<_> = match kind {
                    0 => {
                        if palace(side, source) {
                            DIRS.iter().map(|&(a, b)| (a, b, 0, 0)).collect()
                        } else {
                            vec![]
                        }
                    }
                    1 => [(-1, -1), (1, -1), (-1, 1), (1, 1)]
                        .iter()
                        .map(|&(a, b)| (a, b, 0, 0))
                        .collect(),
                    2 => ELEPHANT.to_vec(),
                    3 => HORSE.to_vec(),
                    6 => {
                        let forward = if side == Color::Red { -1 } else { 1 };
                        let mut v = vec![(0, forward, 0, 0)];
                        if if side == Color::Red { r <= 4 } else { r >= 5 } {
                            v.extend([(-1, 0, 0, 0), (1, 0, 0, 0)]);
                        }
                        v
                    }
                    _ => vec![],
                };
                for (ordinal, (df, dr, gf, gr)) in candidates.into_iter().enumerate() {
                    let Some(target) = sq(f + df, r + dr) else {
                        continue;
                    };
                    if (kind == 0 || kind == 1) && !palace(side, target) {
                        continue;
                    }
                    if kind == 2
                        && if side == Color::Red {
                            target / 9 < 5
                        } else {
                            target / 9 > 4
                        }
                    {
                        continue;
                    }
                    let gate = if gf != 0 || gr != 0 {
                        sq(f + gf, r + gr)
                    } else {
                        None
                    };
                    g.dependencies[class][source] |= 1u128 << target;
                    if let Some(gate) = gate {
                        g.dependencies[class][source] |= 1u128 << gate;
                        let gates = &mut g.gates[class * 90 + source];
                        if let Some((_, targets)) = gates
                            .iter_mut()
                            .find(|(square, _)| *square as usize == gate)
                        {
                            *targets |= 1u128 << target;
                        } else {
                            gates.push((gate as u8, 1u128 << target));
                        }
                    } else {
                        g.unblocked[class][source] |= 1u128 << target;
                    }
                    steps.push((target, gate, ordinal));
                }
                g.steps[class * 90 + source] = steps;
            }
        }
        g
    })
}

impl Board {
    pub fn new(position: &Position) -> Self {
        let mut board = Self {
            cells: [None; 90],
            occupied: 0,
            colors: [0; 2],
            kings: [0; 2],
        };
        for source in position.occupied_squares() {
            board.put(source, position.piece_at(source));
        }
        board
    }
    fn put(&mut self, square: usize, piece: Option<Piece>) {
        self.occupied &= !(1u128 << square);
        for color in &mut self.colors {
            *color &= !(1u128 << square);
        }
        self.cells[square] = piece;
        if let Some(piece) = piece {
            self.occupied |= 1u128 << square;
            self.colors[color_index(piece.color)] |= 1u128 << square;
            if piece.kind == PieceKind::General {
                self.kings[color_index(piece.color)] = square;
            }
        }
    }
    pub fn after(mut self, mv: Move) -> Self {
        let piece = self.cells[mv.from as usize].expect("occupied move source");
        let (from, to) = (1u128 << mv.from, 1u128 << mv.to);
        self.cells[mv.from as usize] = None;
        self.cells[mv.to as usize] = Some(piece);
        self.occupied = (self.occupied & !from) | to;
        let color = color_index(piece.color);
        self.colors[color] = (self.colors[color] & !from) | to;
        self.colors[1 - color] &= !to;
        if piece.kind == PieceKind::General {
            self.kings[color] = mv.to as usize;
        }
        self
    }
    pub fn changed_contents(&self, other: &Self) -> u128 {
        bits(self.occupied | other.occupied).fold(0, |mask, source| {
            if self.cells[source] == other.cells[source] {
                mask
            } else {
                mask | (1u128 << source)
            }
        })
    }
    fn class(&self, source: usize) -> usize {
        let piece = self.cells[source].unwrap();
        color_index(piece.color) * 7 + piece_kind_index(piece.kind)
    }
    pub fn blockers(&self, source: usize, direction: usize) -> u128 {
        self.occupied & geometry().rays[source][direction]
    }
    fn nearest(mask: u128, direction: usize) -> usize {
        if direction == 1 || direction == 3 {
            mask.trailing_zeros() as usize
        } else {
            127 - mask.leading_zeros() as usize
        }
    }
    #[inline]
    fn ray_control(&self, source: usize, direction: usize, cannon: bool) -> u128 {
        let g = geometry();
        let ray = g.rays[source][direction];
        let blockers = self.occupied & ray;
        if blockers == 0 {
            return if cannon { 0 } else { ray };
        }
        let first = Self::nearest(blockers, direction);
        let beyond = g.rays[first][direction];
        if !cannon {
            return ray & !beyond;
        }
        let rest = blockers & beyond;
        if rest == 0 {
            beyond
        } else {
            beyond & !g.rays[Self::nearest(rest, direction)][direction]
        }
    }
    /// 源棋子未改变时，只修补占位变化涉及的射线或门控；目标换色不改变几何控制。
    fn controls_changed(&self, source: usize, previous: u128, occupancy_changes: u128) -> u128 {
        let Some(piece) = self.cells[source] else {
            return 0;
        };
        match piece.kind {
            PieceKind::Rook | PieceKind::Cannon => {
                let mut mask = previous;
                for (direction, &ray) in geometry().rays[source].iter().enumerate() {
                    if ray & occupancy_changes != 0 {
                        mask = (mask & !ray)
                            | self.ray_control(source, direction, piece.kind == PieceKind::Cannon);
                    }
                }
                mask
            }
            PieceKind::Horse | PieceKind::Elephant => {
                let mut mask = previous;
                for &(gate, targets) in &geometry().gates[self.class(source) * 90 + source] {
                    if occupancy_changes & (1u128 << gate) != 0 {
                        if self.occupied & (1u128 << gate) == 0 {
                            mask |= targets;
                        } else {
                            mask &= !targets;
                        }
                    }
                }
                mask
            }
            // 飞将的终点随敌将移动，不能只按占用位是否变化处理。
            PieceKind::General => self.controls(source),
            _ => previous,
        }
    }
    pub fn controls(&self, source: usize) -> u128 {
        let Some(piece) = self.cells[source] else {
            return 0;
        };
        let class = self.class(source);
        let kind = class % 7;
        let g = geometry();
        let mut mask = g.unblocked[class][source];
        for &(gate, targets) in &g.gates[class * 90 + source] {
            if self.occupied & (1u128 << gate) == 0 {
                mask |= targets;
            }
        }
        if kind == 4 || kind == 5 {
            for d in 0..4 {
                mask |= self.ray_control(source, d, kind == 5);
            }
        }
        if kind == 0 {
            let target = self.kings[color_index(piece.color.opposite())];
            if self.cells[target]
                .is_some_and(|p| p.kind == PieceKind::General && p.color != piece.color)
                && source % 9 == target % 9
                && between(source, target) & self.occupied == 0
            {
                mask |= 1u128 << target;
            }
        }
        mask
    }
    fn constraints(&self, source: usize) -> u128 {
        let Some(piece) = self.cells[source] else {
            return 0;
        };
        if !matches!(
            piece.kind,
            PieceKind::Rook | PieceKind::Cannon | PieceKind::General
        ) {
            return 0;
        }
        let king = self.kings[color_index(piece.color.opposite())];
        if source % 9 != king % 9 && (source / 9 != king / 9 || piece.kind == PieceKind::General) {
            return 0;
        }
        let path = between(source, king) & self.occupied;
        if path.count_ones()
            == if piece.kind == PieceKind::Cannon {
                2
            } else {
                1
            }
        {
            path & self.colors[color_index(piece.color.opposite())]
        } else {
            0
        }
    }
}

pub(crate) fn between(a: usize, b: usize) -> u128 {
    if a % 9 != b % 9 && a / 9 != b / 9 {
        return 0;
    }
    let step = if a / 9 == b / 9 { 1 } else { 9 };
    let (lo, hi) = (a.min(b), a.max(b));
    let mut mask = 0;
    let mut s = lo + step;
    while s < hi {
        mask |= 1u128 << s;
        s += step;
    }
    mask
}

fn changed_royals(before: &Board, after: &Board) -> u128 {
    before
        .kings
        .iter()
        .zip(after.kings)
        .fold(0, |mask, (&a, b)| {
            if a == b {
                mask
            } else {
                mask | (1u128 << a) | (1u128 << b)
            }
        })
}

#[derive(Clone)]
pub(crate) struct Graph {
    pub board: Board,
    pub controls: [u128; 90],
    pub incoming: [u128; 90],
    reverse_dependencies: [u128; 90],
    constraint_sources: SmallVec<[(u8, u128); 8]>,
    constrained: u128,
}

impl Graph {
    pub fn new(board: Board) -> Self {
        let mut graph = Self {
            board,
            controls: [0; 90],
            incoming: [0; 90],
            reverse_dependencies: [0; 90],
            constraint_sources: SmallVec::new(),
            constrained: 0,
        };
        for source in bits(board.occupied) {
            graph.controls[source] = board.controls(source);
            for target in bits(graph.controls[source]) {
                graph.incoming[target] |= 1u128 << source;
            }
            for target in bits(geometry().dependencies[board.class(source)][source]) {
                graph.reverse_dependencies[target] |= 1u128 << source;
            }
            let constraints = board.constraints(source);
            if constraints != 0 {
                graph.constraint_sources.push((source as u8, constraints));
            }
            graph.constrained |= constraints;
        }
        graph
    }
    /// 保留未改变的源、入边和逆依赖；根局面、小残局或对照验证可选择完整构建。
    pub fn updated(&self, board: Board, changed: u128) -> Self {
        let mut next = self.clone();
        next.board = board;
        let occupancy_changes = self.board.occupied ^ board.occupied;
        let sources = self.estimated_dirty_sources(changed | changed_royals(&self.board, &board))
            & (self.board.occupied | board.occupied);
        for source in bits(sources) {
            let content_changed = self.board.cells[source] != board.cells[source];
            let controls = if content_changed {
                board.controls(source)
            } else {
                board.controls_changed(source, self.controls[source], occupancy_changes)
            };
            for target in bits(self.controls[source] ^ controls) {
                next.incoming[target] ^= 1u128 << source;
            }
            next.controls[source] = controls;
            let constraints = board.constraints(source);
            if let Some(index) = next
                .constraint_sources
                .iter()
                .position(|&(s, _)| s as usize == source)
            {
                if constraints == 0 {
                    next.constraint_sources.swap_remove(index);
                } else {
                    next.constraint_sources[index].1 = constraints;
                }
            } else if constraints != 0 {
                next.constraint_sources.push((source as u8, constraints));
            }
            if content_changed {
                let old = self.board.cells[source].map_or(0, |_| {
                    geometry().dependencies[self.board.class(source)][source]
                });
                let new = board.cells[source]
                    .map_or(0, |_| geometry().dependencies[board.class(source)][source]);
                for target in bits(old ^ new) {
                    next.reverse_dependencies[target] ^= 1u128 << source;
                }
            }
        }
        next.constrained = next
            .constraint_sources
            .iter()
            .fold(0, |mask, &(_, part)| mask | part);
        next
    }
    pub fn heap_bytes(&self) -> usize {
        if self.constraint_sources.spilled() {
            self.constraint_sources.capacity() * std::mem::size_of::<(u8, u128)>()
        } else {
            0
        }
    }
    pub fn origins(&self, target: usize, color: Color) -> u128 {
        self.incoming[target] & self.board.colors[color_index(color)]
    }
    pub fn enemies(&self, source: usize) -> u128 {
        let Some(piece) = self.board.cells[source] else {
            return 0;
        };
        self.controls[source] & self.board.colors[color_index(piece.color.opposite())]
    }
    pub fn guards(&self, target: usize) -> u128 {
        let Some(piece) = self.board.cells[target] else {
            return 0;
        };
        self.origins(target, piece.color) & !(1u128 << target)
    }
    pub fn constrained(&self, source: usize) -> bool {
        self.constrained & (1u128 << source) != 0
    }
    pub fn patch(&self, mv: Move) -> Patch {
        let board = self.board.after(mv);
        let changed = (1u128 << mv.from) | (1u128 << mv.to);
        let potential = changed
            | self.reverse_dependencies[mv.from as usize]
            | self.reverse_dependencies[mv.to as usize];
        let mut records = SmallVec::new();
        let occupancy_changes = self.board.occupied ^ board.occupied;
        let mut dirty = 0;
        for source in bits(potential & (self.board.occupied | board.occupied | changed)) {
            let old = self.controls[source];
            let mask = if changed & (1u128 << source) != 0 {
                board.controls(source)
            } else {
                board.controls_changed(source, old, occupancy_changes)
            };
            if changed & (1u128 << source) != 0 || old != mask || old & changed != 0 {
                records.push((source, mask));
                dirty |= 1u128 << source;
            }
        }
        Patch {
            board,
            dirty,
            records,
        }
    }
    pub fn estimated_dirty_sources(&self, changed: u128) -> u128 {
        bits(changed).fold(changed, |mask, square| {
            mask | self.reverse_dependencies[square]
        })
    }
    fn vacancy(&self, from: usize) -> Vacancy {
        let mut board = self.board;
        board.put(from, None);
        let potential = (1u128 << from) | self.reverse_dependencies[from];
        let mut dirty = 0;
        let mut records = SmallVec::new();
        for source in bits(potential & self.board.occupied) {
            let old = self.controls[source];
            let mask = if source == from {
                0
            } else {
                board.controls_changed(source, old, 1u128 << from)
            };
            if source == from || old != mask || old & (1u128 << from) != 0 {
                records.push((source, mask));
                dirty |= 1u128 << source;
            }
        }
        Vacancy { dirty, records }
    }
    fn patch_vacancy(&self, mv: Move, vacancy: &Vacancy) -> Patch {
        let board = self.board.after(mv);
        let to = mv.to as usize;
        let potential = (1u128 << to) | self.reverse_dependencies[to];
        let occupancy_changes = (1u128 << to) & !self.board.occupied;
        let mut records = vacancy.records.clone();
        let mut dirty = vacancy.dirty;
        for source in bits(potential & (self.board.occupied | board.occupied)) {
            let slot = records.iter().position(|(s, _)| *s == source);
            let previous = slot.map_or(self.controls[source], |i| records[i].1);
            let mask = if source == to || source == mv.from as usize {
                board.controls(source)
            } else {
                board.controls_changed(source, previous, occupancy_changes)
            };
            if let Some(i) = slot {
                records[i].1 = mask;
            } else if mask != self.controls[source]
                || source == to
                || self.controls[source] & (1u128 << to) != 0
            {
                records.push((source, mask));
                dirty |= 1u128 << source;
            }
        }
        Patch {
            board,
            dirty,
            records,
        }
    }
}

struct Vacancy {
    dirty: u128,
    records: SmallVec<[(usize, u128); 8]>,
}
pub(crate) struct Patch {
    pub board: Board,
    pub dirty: u128,
    pub records: SmallVec<[(usize, u128); 8]>,
}
impl Patch {
    fn all_origins(&self, graph: &Graph, target: usize) -> u128 {
        let mut origins = graph.incoming[target] & !self.dirty;
        for &(source, mask) in &self.records {
            if mask & (1u128 << target) != 0 {
                origins |= 1u128 << source;
            }
        }
        origins
    }
    pub fn origins(&self, graph: &Graph, target: usize, color: Color) -> u128 {
        self.all_origins(graph, target) & self.board.colors[color_index(color)]
    }
    pub fn controls(&self, graph: &Graph, source: usize) -> u128 {
        self.records
            .iter()
            .find(|&&(s, _)| s == source)
            .map_or(graph.controls[source], |&(_, mask)| mask)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FeatureRows {
    pub relations: Vec<u32>,
    pub motifs: Vec<u32>,
}

pub(crate) fn relation_rows(graph: &Graph, perspective: Color, sources: u128) -> Vec<u32> {
    let mut rows =
        Vec::with_capacity(((graph.board.occupied & sources).count_ones() as usize * 6).min(256));
    for source in bits(graph.board.occupied & sources) {
        append_relations(
            &mut rows,
            graph,
            perspective,
            source,
            graph.controls[source] & graph.board.occupied,
        );
    }
    rows
}

fn append_relations(
    rows: &mut impl Extend<u32>,
    graph: &Graph,
    perspective: Color,
    source: usize,
    targets: u128,
) {
    if targets != 0 {
        let piece = graph.board.cells[source].unwrap();
        let from = canonical_square(perspective, source);
        let source_class = piece_absolute_feature_index(perspective, piece);
        for target in bits(targets) {
            let victim = graph.board.cells[target].unwrap();
            let to = canonical_square(perspective, target);
            let dr = to as i32 / 9 - from as i32 / 9;
            let df = to as i32 % 9 - from as i32 % 9;
            let delta = ((dr + 9) * 17 + df + 8) as usize;
            let core = ((((source_class / 7 * 7 + source_class % 7) * 2
                + usize::from(victim.color != piece.color))
                * 7
                + piece_kind_index(victim.kind))
                * 323)
                + delta;
            rows.extend([
                core as u32,
                (RELATION_CORE_ROWS + source_class * 90 + from) as u32,
                (RELATION_CORE_ROWS
                    + 1260
                    + piece_absolute_feature_index(perspective, victim) * 90
                    + to) as u32,
            ]);
        }
    }
}

pub(crate) fn relation_changes(
    before: &Graph,
    after: &Graph,
    perspective: Color,
    sources: u128,
    changed: u128,
) -> (SmallVec<[u32; 64]>, SmallVec<[u32; 64]>) {
    let (mut removed, mut added) = (SmallVec::new(), SmallVec::new());
    for source in bits(sources) {
        let old = before.controls[source] & before.board.occupied;
        let new = after.controls[source] & after.board.occupied;
        let (old, new) = if before.board.cells[source] == after.board.cells[source] {
            (old & (!new | changed), new & (!old | changed))
        } else {
            (old, new)
        };
        append_relations(&mut removed, before, perspective, source, old);
        append_relations(&mut added, after, perspective, source, new);
    }
    (removed, added)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Motif {
    pub family: u8,
    pub root: u8,
    pub b: u8,
    pub c: u8,
    pub geometry: u8,
}
const EMPTY: u8 = 90;
fn distance(a: usize, b: usize) -> usize {
    (a / 9).abs_diff(b / 9) + (a % 9).abs_diff(b % 9)
}
fn d8(n: usize) -> usize {
    match n {
        0..=4 => n,
        5..=6 => 5,
        7..=9 => 6,
        _ => 7,
    }
}
fn d4(n: usize) -> usize {
    match n {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        _ => 3,
    }
}
fn count4(n: usize) -> usize {
    n.min(3)
}
fn amount(n: usize) -> usize {
    match n {
        0 | 1 => 0,
        2 => 1,
        3..=4 => 2,
        _ => 3,
    }
}
fn dir_relation(root: usize, b: usize, c: usize) -> usize {
    let (x, y) = (
        b as i32 % 9 - root as i32 % 9,
        b as i32 / 9 - root as i32 / 9,
    );
    let (u, v) = (
        c as i32 % 9 - root as i32 % 9,
        c as i32 / 9 - root as i32 / 9,
    );
    let dot = x * u + y * v;
    if x * v == y * u {
        usize::from(dot < 0)
    } else if dot == 0 {
        2
    } else {
        3
    }
}
fn canonical_direction(side: Color, d: usize) -> usize {
    if side == Color::Red { d } else { d ^ 1 }
}
fn ordered(graph: &Graph, a: usize, b: usize, perspective: Color) -> (usize, usize) {
    if (
        piece_kind_index(graph.board.cells[a].unwrap().kind),
        canonical_square(perspective, a),
    ) < (
        piece_kind_index(graph.board.cells[b].unwrap().kind),
        canonical_square(perspective, b),
    ) {
        (a, b)
    } else {
        (b, a)
    }
}

/// 先确认参与者共同身份，再编码；只提取指定根，供增量更新复用。
pub(crate) fn motifs(graph: &Graph, roots: u128, perspective: Color) -> Vec<Motif> {
    if roots & graph.board.occupied == 0 {
        return Vec::new();
    }
    let mut events = Vec::with_capacity(128);
    let board = &graph.board;
    let mut emit = |family: usize, root: usize, b: usize, c: usize, geometry: usize| {
        debug_assert!(family < 12 && geometry < 128);
        events.push(Motif {
            family: family as u8,
            root: root as u8,
            b: b as u8,
            c: c as u8,
            geometry: geometry as u8,
        });
    };
    for root in bits(board.occupied & roots) {
        let piece = board.cells[root].unwrap();
        let side = piece.color;
        let kind = piece_kind_index(piece.kind);
        let own = board.colors[color_index(side)];
        let enemy = board.colors[color_index(side.opposite())];
        if kind == 4 || kind == 5 {
            for direction in 0..4 {
                let mut mask = board.blockers(root, direction);
                let mut blockers = [90; 3];
                for b in &mut blockers {
                    if mask == 0 {
                        break;
                    }
                    *b = Board::nearest(mask, direction);
                    mask &= !(1u128 << *b);
                }
                let d = canonical_direction(perspective, direction);
                if kind == 4 && blockers[1] != 90 {
                    for ordinal in 1..=2 {
                        let target = blockers[ordinal];
                        if target == 90 {
                            continue;
                        }
                        emit(
                            0,
                            root,
                            blockers[0],
                            target,
                            (d * 8 + d8(distance(root, target))) * 4 + ordinal,
                        );
                    }
                }
                if kind == 5 && blockers[1] != 90 {
                    emit(
                        1,
                        root,
                        blockers[0],
                        blockers[1],
                        (d * 4 + d4(distance(root, blockers[0]) - 1)) * 8
                            + d8(distance(blockers[0], blockers[1]) - 1),
                    );
                }
                if let Some(front) = board.cells.get(blockers[0]).copied().flatten() {
                    if front.color == side
                        && matches!(front.kind, PieceKind::Rook | PieceKind::Cannon)
                        && blockers[1] != 90
                        && board.cells[blockers[1]].unwrap().color != side
                        && roots & (1u128 << blockers[0]) != 0
                    {
                        emit(
                            8,
                            blockers[0],
                            root,
                            blockers[1],
                            (d * 4 + d4(distance(root, blockers[0]) - 1)) * 8
                                + d8(distance(blockers[0], blockers[1]) - 1),
                        );
                    }
                }
            }
        }
        if kind == 2 || kind == 3 {
            for &(target, gate, ordinal) in &geometry().steps[board.class(root) * 90 + root] {
                if let Some(gate) = gate
                    && board.cells[gate].is_some()
                    && board.cells[target].is_some()
                {
                    let target_canonical = canonical_square(perspective, target);
                    let f = target_canonical % 9;
                    let zone = usize::from(target_canonical / 9 >= 5) * 4
                        + match f {
                            4 => 0,
                            3 | 5 => 1,
                            2 | 6 => 2,
                            _ => 3,
                        };
                    let direction = if perspective == Color::Red {
                        ordinal
                    } else if kind == 3 {
                        [3, 2, 1, 0, 7, 6, 5, 4][ordinal]
                    } else {
                        3 - ordinal
                    };
                    emit(2, root, gate, target, direction * 8 + zone);
                }
            }
        }
        let targets: SmallVec<[usize; 8]> = bits(graph.controls[root] & enemy).collect();
        for (i, &a) in targets.iter().enumerate() {
            for &b in &targets[i + 1..] {
                let (a, b) = ordered(graph, a, b, perspective);
                let state =
                    usize::from(graph.guards(a) != 0) | usize::from(graph.guards(b) != 0) << 1;
                emit(
                    3,
                    root,
                    a,
                    b,
                    (dir_relation(root, a, b) * 8 + d8(distance(root, a).max(distance(root, b))))
                        * 4
                        + state,
                );
            }
        }
        let attackers: SmallVec<[usize; 16]> = bits(graph.origins(root, side.opposite())).collect();
        for (i, &a) in attackers.iter().enumerate() {
            for &b in &attackers[i + 1..] {
                let (a, b) = ordered(graph, a, b, perspective);
                emit(
                    4,
                    root,
                    a,
                    b,
                    (dir_relation(root, a, b) * 8 + d8(distance(root, a).max(distance(root, b))))
                        * 4
                        + count4(graph.guards(root).count_ones() as usize),
                );
            }
        }
        let guarded: SmallVec<[usize; 8]> =
            bits(graph.controls[root] & own & !(1u128 << root)).collect();
        for &attacker in &attackers {
            for &target in &guarded {
                let state =
                    usize::from(graph.origins(target, side.opposite()) & !(1u128 << attacker) != 0)
                        | usize::from(graph.guards(target) & !(1u128 << root) != 0) << 1;
                emit(
                    5,
                    root,
                    attacker,
                    target,
                    (dir_relation(root, attacker, target) * 8
                        + d8(distance(root, attacker).max(distance(root, target))))
                        * 4
                        + state,
                );
            }
        }
        for (i, &a) in guarded.iter().enumerate() {
            for &b in &guarded[i + 1..] {
                let (a, b) = ordered(graph, a, b, perspective);
                let state = usize::from(graph.origins(a, side.opposite()) != 0)
                    | usize::from(graph.origins(b, side.opposite()) != 0) << 1;
                emit(
                    6,
                    root,
                    a,
                    b,
                    (dir_relation(root, a, b) * 8 + d8(distance(root, a).max(distance(root, b))))
                        * 4
                        + state,
                );
            }
        }
        for &attacker in &attackers {
            for defender in bits(graph.guards(root)) {
                let constraints = usize::from(graph.constrained(attacker))
                    | usize::from(graph.constrained(defender)) << 1;
                emit(
                    7,
                    root,
                    attacker,
                    defender,
                    (constraints * 4 + count4(attackers.len())) * 4
                        + count4(graph.guards(root).count_ones() as usize),
                );
            }
        }
        if matches!(
            piece.kind,
            PieceKind::Rook | PieceKind::Cannon | PieceKind::General
        ) {
            let king = board.kings[color_index(side.opposite())];
            if root % 9 == king % 9 || (root / 9 == king / 9 && piece.kind != PieceKind::General) {
                let path = between(root, king) & board.occupied;
                let n = path.count_ones() as usize;
                let direction = if root % 9 == king % 9 {
                    usize::from(king > root)
                } else {
                    2 + usize::from(king > root)
                };
                let first = if path == 0 {
                    90
                } else {
                    Board::nearest(path, direction)
                };
                emit(
                    9,
                    root,
                    first,
                    king,
                    (canonical_direction(perspective, direction) * 8 + n.min(4)) * 4
                        + d4(distance(root, king)),
                );
            }
        }
    }
    // M8/M11的根是门控子；只访问与这些根存在潜在几何依赖的同色源。
    let pivots = if roots & board.occupied == board.occupied {
        board.occupied
    } else {
        board.colors.iter().fold(0, |sources, &color| {
            sources | (graph.estimated_dirty_sources(roots & color) & color)
        })
    };
    for rear in bits(pivots & !roots) {
        let p = board.cells[rear].unwrap();
        if !matches!(p.kind, PieceKind::Rook | PieceKind::Cannon) {
            continue;
        }
        for d in 0..4 {
            let mask = board.blockers(rear, d);
            if mask == 0 {
                continue;
            }
            let front = Board::nearest(mask, d);
            let fp = board.cells[front].unwrap();
            if roots & (1u128 << front) == 0
                || fp.color != p.color
                || !matches!(fp.kind, PieceKind::Rook | PieceKind::Cannon)
            {
                continue;
            }
            let rest = mask & !(1u128 << front);
            if rest == 0 {
                continue;
            }
            let target = Board::nearest(rest, d);
            if board.cells[target].unwrap().color != p.color {
                emit(
                    8,
                    front,
                    rear,
                    target,
                    (canonical_direction(perspective, d) * 4 + d4(distance(rear, front) - 1)) * 8
                        + d8(distance(front, target) - 1),
                );
            }
        }
    }
    // M11：同一门控子兼有第二威胁，A/T/B/U 身份均在这里验证。
    for source in bits(pivots) {
        let piece = board.cells[source].unwrap();
        let kind = piece_kind_index(piece.kind);
        let mut gates: SmallVec<[(usize, usize, usize); 12]> = SmallVec::new();
        if kind == 4 || kind == 5 {
            for d in 0..4 {
                let mask = board.blockers(source, d);
                let mut squares: SmallVec<[usize; 3]> = SmallVec::new();
                let mut m = mask;
                for _ in 0..3 {
                    if m == 0 {
                        break;
                    }
                    let s = Board::nearest(m, d);
                    m &= !(1u128 << s);
                    squares.push(s);
                }
                if kind == 4 && squares.len() >= 2 {
                    gates.push((squares[0], squares[1], 0));
                }
                if kind == 5 && squares.len() == 3 {
                    for gate in &squares[..2] {
                        gates.push((*gate, squares[2], 1));
                    }
                }
            }
        } else if kind == 2 || kind == 3 {
            for &(target, gate, _) in &geometry().steps[board.class(source) * 90 + source] {
                if let Some(gate) = gate
                    && board.cells[gate].is_some()
                    && board.cells[target].is_some()
                {
                    gates.push((gate, target, if kind == 3 { 2 } else { 3 }));
                }
            }
        }
        for (gate, target, gate_kind) in gates {
            if roots & (1u128 << gate) == 0
                || board.cells[gate].unwrap().color != piece.color
                || board.cells[target].unwrap().color == piece.color
            {
                continue;
            }
            for second in bits(graph.enemies(gate) & !(1u128 << target)) {
                emit(
                    11,
                    gate,
                    source,
                    target,
                    (piece_kind_index(board.cells[second].unwrap().kind) * 4 + gate_kind) * 2
                        + usize::from(board.cells[target].unwrap().kind == PieceKind::General),
                );
            }
        }
    }
    events
}

pub(crate) fn encode_motifs(graph: &Graph, events: &[Motif], perspective: Color) -> Vec<u32> {
    let mut rows = Vec::with_capacity(events.len() * 3);
    emit_motif_rows(graph, events, perspective, |_, factor_rows| {
        rows.extend(factor_rows)
    });
    rows
}

pub(crate) fn emit_motif_rows(
    graph: &Graph,
    events: &[Motif],
    perspective: Color,
    mut emit: impl FnMut(usize, [u32; 3]),
) {
    let mut keys = Vec::with_capacity(events.len());
    for event in events {
        let root = event.root as usize;
        let p = graph.board.cells[root].unwrap();
        let mut variant = 0;
        let roles = [event.b, event.c].map(|square| {
            if square == EMPTY {
                7
            } else {
                piece_kind_index(graph.board.cells[square as usize].unwrap().kind)
            }
        });
        for (i, square) in [event.b, event.c].into_iter().enumerate() {
            if square != EMPTY && graph.board.cells[square as usize].unwrap().color != p.color {
                variant |= 1 << i;
            }
        }
        keys.push([
            event.family,
            event.root,
            roles[0] as u8,
            roles[1] as u8,
            variant as u8,
            event.geometry,
        ]);
    }
    keys.sort_unstable();
    let mut i = 0;
    while i < keys.len() {
        let key = keys[i];
        let mut end = i + 1;
        while end < keys.len() && keys[end] == key {
            end += 1;
        }
        let [family, root, b, c, variant, geo] = key.map(usize::from);
        let piece = graph.board.cells[root].unwrap();
        let owner = usize::from(piece.color != perspective);
        let kind = piece_kind_index(piece.kind);
        let core = (((((family * 2 + owner) * 7 + kind) * 8 + b) * 8 + c) * 4) + variant;
        let geometry = MOTIF_CORE_ROWS + (((family * 2 + owner) * 128 + geo) * 4 + amount(end - i));
        let anchor = MOTIF_CORE_ROWS
            + MOTIF_GEOMETRY_ROWS
            + (family * 2 + owner) * 90
            + canonical_square(perspective, root);
        emit(root, [core as u32, geometry as u32, anchor as u32]);
        i = end;
    }
}

pub(crate) fn extract(graph: &Graph, perspective: Color, roots: u128) -> FeatureRows {
    FeatureRows {
        relations: relation_rows(graph, perspective, roots),
        motifs: encode_motifs(graph, &motifs(graph, roots, perspective), perspective),
    }
}

pub(crate) fn king_context(graph: &Graph, perspective: Color) -> ([f32; 14], Vec<u32>) {
    let mut context = [0.; 14];
    let mut seals = Vec::new();
    for (i, side) in [perspective, perspective.opposite()]
        .into_iter()
        .enumerate()
    {
        let king = graph.board.kings[color_index(side)];
        let checkers = graph.origins(king, side.opposite());
        context[i * 7] = f32::from(checkers != 0);
        context[i * 7 + 1] = count4(checkers.count_ones() as usize) as f32 / 3.;
        for (d, (df, dr)) in DIRS.into_iter().enumerate() {
            let Some(target) = sq(king as i32 % 9 + df, king as i32 / 9 + dr) else {
                continue;
            };
            if !palace(side, target) || graph.board.cells[target].is_some_and(|p| p.color == side) {
                continue;
            }
            let patch = graph.patch(Move::new(king, target));
            let origins = patch.origins(graph, target, side.opposite());
            let oriented_d = canonical_direction(perspective, d);
            context[i * 7 + 2 + oriented_d] = f32::from(origins == 0);
            if checkers != 0 {
                let target_side = canonical_square(side, target);
                let local = (target_side / 9 - 7) * 3 + (target_side % 9 - 3);
                for checker in bits(checkers) {
                    for sealer in bits(origins & !(1u128 << checker)) {
                        seals.push(Motif {
                            family: 10,
                            root: king as u8,
                            b: checker as u8,
                            c: sealer as u8,
                            geometry: (local * 2 + usize::from(graph.board.cells[target].is_some()))
                                as u8,
                        });
                    }
                }
            }
        }
        context[i * 7 + 6] = context[i * 7 + 2..i * 7 + 6].iter().sum::<f32>() / 4.;
    }
    (context, encode_motifs(graph, &seals, perspective))
}

#[derive(Clone, Debug)]
pub(crate) struct MoveFeatures {
    pub local: [f32; MOVE_LOCAL_SIZE],
    pub joint: u32,
    pub events: SmallVec<[u32; 24]>,
    pub moved: usize,
    pub captured: usize,
    pub from_feature: usize,
    pub to_feature: usize,
    pub captured_feature: Option<usize>,
}

pub(crate) fn move_features(
    graph: &Graph,
    mv: Move,
    perspective: Color,
    repetition: Option<bool>,
) -> MoveFeatures {
    move_features_with_patch(graph, mv, perspective, repetition, graph.patch(mv))
}

pub(crate) struct MoveFeatureCache<'a> {
    graph: &'a Graph,
    slots: [u8; 90],
    counts: [u8; 90],
    vacancies: SmallVec<[Vacancy; 16]>,
}
impl<'a> MoveFeatureCache<'a> {
    pub fn new(graph: &'a Graph, moves: impl IntoIterator<Item = Move>) -> Self {
        let mut cache = Self {
            graph,
            slots: [u8::MAX; 90],
            counts: [0; 90],
            vacancies: SmallVec::new(),
        };
        for m in moves {
            cache.counts[m.from as usize] = cache.counts[m.from as usize].saturating_add(1);
        }
        cache
    }
    pub fn features(
        &mut self,
        mv: Move,
        perspective: Color,
        repetition: Option<bool>,
    ) -> MoveFeatures {
        let from = mv.from as usize;
        if self.counts[from] < 4 {
            return move_features(self.graph, mv, perspective, repetition);
        }
        if self.slots[from] == u8::MAX {
            self.slots[from] = self.vacancies.len() as u8;
            self.vacancies.push(self.graph.vacancy(from));
        }
        let patch = self
            .graph
            .patch_vacancy(mv, &self.vacancies[self.slots[from] as usize]);
        move_features_with_patch(self.graph, mv, perspective, repetition, patch)
    }
}

fn move_features_with_patch(
    graph: &Graph,
    mv: Move,
    perspective: Color,
    repetition: Option<bool>,
    patch: Patch,
) -> MoveFeatures {
    let from = mv.from as usize;
    let to = mv.to as usize;
    let moved = graph.board.cells[from].unwrap();
    let side = moved.color;
    let kind = piece_kind_index(moved.kind);
    let captured = graph.board.cells[to].map_or(7, |p| piece_kind_index(p.kind));
    let destination_origins = patch.all_origins(graph, to);
    let hostile = destination_origins & patch.board.colors[color_index(side.opposite())];
    let friendly = destination_origins & patch.board.colors[color_index(side)] & !(1u128 << to);
    let king = patch.board.kings[color_index(side.opposite())];
    let checkers = patch.origins(graph, king, side);
    let after_targets =
        patch.controls(graph, to) & patch.board.colors[color_index(side.opposite())];
    let before_targets = graph.enemies(from);
    let enemy_count = count4(hostile.count_ones() as usize);
    let own_count = count4(friendly.count_ones() as usize);
    let local = [
        f32::from(captured != 7),
        f32::from(checkers != 0),
        count4(checkers.count_ones() as usize) as f32 / 3.,
        enemy_count as f32 / 3.,
        own_count as f32 / 3.,
        count4(graph.origins(from, side.opposite()).count_ones() as usize) as f32 / 3.,
        count4(graph.guards(from).count_ones() as usize) as f32 / 3.,
        count4((after_targets & !before_targets).count_ones() as usize) as f32 / 3.,
        count4((before_targets & !after_targets).count_ones() as usize) as f32 / 3.,
        f32::from(checkers & !(1u128 << to) != 0),
        f32::from(repetition.unwrap_or(false)),
        f32::from(repetition.is_some()),
    ];
    // 同一候选的走子棋种固定，192个局部键即可准确计数，避免排序重复事件。
    let mut event_counts = [0u16; MOTIF_FAMILIES * 8 * 2];
    let mut active_events = [0u128; 2];
    let mut emit = |family: usize, target_kind: usize, added: bool| {
        let key = (family * 8 + target_kind) * 2 + usize::from(added);
        event_counts[key] += 1;
        active_events[key / 128] |= 1u128 << (key % 128);
    };
    for &(source, new_controls) in &patch.records {
        let old_piece = graph.board.cells[source];
        let new_piece = patch.board.cells[source];
        if !old_piece.is_some_and(|p| p.color == side)
            && !new_piece.is_some_and(|p| p.color == side)
        {
            continue;
        }
        let old = graph.controls[source];
        let old_targets = old & graph.board.occupied;
        let new_targets = new_controls & patch.board.occupied;
        let changed = (1u128 << from) | (1u128 << to);
        let (old_changed, new_changed) = if source == from || source == to {
            (old_targets, new_targets)
        } else {
            (
                old_targets & (!new_targets | changed),
                new_targets & (!old_targets | changed),
            )
        };
        let new_guarded = new_controls & patch.board.colors[color_index(side)];
        let old_guarded = old & graph.board.colors[color_index(side)];
        let after_attacked = if source == to {
            hostile != 0
        } else {
            new_piece.is_some_and(|p| p.color == side)
                && new_changed & new_guarded != 0
                && patch.origins(graph, source, side.opposite()) != 0
        };
        for (active, piece, added) in [
            (old_changed, old_piece, false),
            (new_changed, new_piece, true),
        ] {
            let Some(piece) = piece else { continue };
            if piece.color != side {
                continue;
            }
            for target in bits(active) {
                let Some(victim) = (if added {
                    patch.board.cells[target]
                } else {
                    graph.board.cells[target]
                }) else {
                    continue;
                };
                let target_kind = piece_kind_index(victim.kind);
                if piece.kind == PieceKind::Cannon {
                    emit(1, target_kind, added);
                }
                if piece.kind == PieceKind::Rook {
                    emit(0, target_kind, added);
                }
                if matches!(piece.kind, PieceKind::Horse | PieceKind::Elephant) {
                    emit(2, target_kind, added);
                }
                if victim.color == side {
                    let attacked = if added {
                        after_attacked
                    } else {
                        graph.origins(source, side.opposite()) != 0
                    };
                    if attacked {
                        emit(5, target_kind, added);
                    }
                    let guarded = if added { new_guarded } else { old_guarded };
                    if guarded.count_ones() >= 2 {
                        emit(6, target_kind, added);
                    }
                }
            }
        }
        if source != from && source != to && new_piece.is_some_and(|p| p.color == side) {
            let old_n = (old & graph.board.colors[color_index(side.opposite())]).count_ones();
            let new_n =
                (new_controls & patch.board.colors[color_index(side.opposite())]).count_ones();
            if old_n < 2 && new_n >= 2 {
                emit(3, 7, true);
            } else if old_n >= 2 && new_n < 2 {
                emit(3, 7, false);
            }
            for target in
                bits(new_controls & patch.board.colors[color_index(side.opposite())] & !old)
            {
                if after_targets & !(1u128 << target) != 0 {
                    emit(
                        11,
                        piece_kind_index(patch.board.cells[target].unwrap().kind),
                        true,
                    );
                }
            }
        }
    }
    let old_n = before_targets.count_ones();
    let new_n = after_targets.count_ones();
    if old_n < 2 && new_n >= 2 {
        emit(3, 7, true);
    } else if old_n >= 2 && new_n < 2 {
        emit(3, 7, false);
    }
    let mut events = SmallVec::new();
    for (word, active) in active_events.into_iter().enumerate() {
        for bit in bits(active) {
            let key = word * 128 + bit;
            let row = ((key / 16 * 7 + kind) * 16 + key % 16) as u32;
            events.extend(std::iter::repeat_n(row, event_counts[key] as usize));
        }
    }
    MoveFeatures {
        local,
        joint: ((((kind * 8 + captured) * 4 + enemy_count) * 4 + own_count) * 2
            + usize::from(checkers != 0)) as u32,
        events,
        moved: kind,
        captured,
        from_feature: kind * 90 + canonical_square(perspective, from),
        to_feature: kind * 90 + canonical_square(perspective, to),
        captured_feature: graph.board.cells[to].map(|p| {
            piece_absolute_feature_index(perspective, p) * 90 + canonical_square(perspective, to)
        }),
    }
}

/// 全部依赖源及旧新关系的二跳闭包。潜在门控源先由完整几何依赖定位。
pub(crate) fn dirty_roots(before: &Graph, after: &Graph, changed: u128) -> (u128, u128) {
    let dependency_changes = changed | changed_royals(&before.board, &after.board);
    let sources = before.estimated_dirty_sources(dependency_changes)
        | after.estimated_dirty_sources(dependency_changes);
    let mut roots = changed;
    let mut targets_changed = changed;
    for source in bits(sources) {
        let old_targets = before.controls[source] & before.board.occupied;
        let new_targets = after.controls[source] & after.board.occupied;
        let mut edge_changed =
            (old_targets ^ new_targets) | ((old_targets | new_targets) & changed);
        if before.board.cells[source] != after.board.cells[source] {
            edge_changed |= old_targets | new_targets;
        }
        if edge_changed != 0 {
            roots |= 1u128 << source;
            targets_changed |= edge_changed;
        }
        for graph in [before, after] {
            let Some(piece) = graph.board.cells[source] else {
                continue;
            };
            let source_changed = before.board.cells[source] != after.board.cells[source];
            if matches!(piece.kind, PieceKind::Rook | PieceKind::Cannon) {
                for d in 0..4 {
                    let ray = geometry().rays[source][d];
                    if !source_changed && ray & changed == 0 {
                        continue;
                    }
                    let mut mask = graph.board.blockers(source, d);
                    let mut blockers = SmallVec::<[usize; 3]>::new();
                    for _ in 0..3 {
                        if mask == 0 {
                            break;
                        }
                        let s = Board::nearest(mask, d);
                        blockers.push(s);
                        mask &= !(1u128 << s);
                    }
                    let observed = if blockers.len() == 3 {
                        ray & !geometry().rays[blockers[2]][d]
                    } else {
                        ray
                    };
                    if !source_changed && observed & changed == 0 {
                        continue;
                    }
                    roots |= 1u128 << source;
                    if blockers.len() >= 2 {
                        let front = graph.board.cells[blockers[0]].unwrap();
                        let second = graph.board.cells[blockers[1]].unwrap();
                        if front.color == piece.color && second.color != piece.color {
                            if piece.kind == PieceKind::Rook
                                || matches!(front.kind, PieceKind::Rook | PieceKind::Cannon)
                            {
                                // M8以前子为根，M11车线以第一门控子为根。
                                roots |= 1u128 << blockers[0];
                            }
                        }
                    }
                    if piece.kind == PieceKind::Cannon
                        && blockers.len() == 3
                        && graph.board.cells[blockers[2]].unwrap().color != piece.color
                    {
                        for &gate in &blockers[..2] {
                            if graph.board.cells[gate].unwrap().color == piece.color {
                                roots |= 1u128 << gate;
                            }
                        }
                    }
                }
            } else if matches!(piece.kind, PieceKind::Horse | PieceKind::Elephant) {
                for &(target, gate, _) in &geometry().steps[graph.board.class(source) * 90 + source]
                {
                    if let Some(gate) = gate
                        && (source_changed || changed & ((1u128 << target) | (1u128 << gate)) != 0)
                        && graph.board.cells[gate].is_some()
                        && graph.board.cells[target].is_some()
                    {
                        roots |= 1u128 << source;
                        if graph.board.cells[gate].unwrap().color == piece.color
                            && graph.board.cells[target].unwrap().color != piece.color
                        {
                            roots |= 1u128 << gate;
                        }
                    }
                }
            }
            if matches!(
                piece.kind,
                PieceKind::Rook | PieceKind::Cannon | PieceKind::General
            ) {
                let king = graph.board.kings[color_index(piece.color.opposite())];
                if (source % 9 == king % 9
                    || (piece.kind != PieceKind::General && source / 9 == king / 9))
                    && (source_changed
                        || dependency_changes & (between(source, king) | (1u128 << king)) != 0)
                {
                    roots |= 1u128 << source;
                }
            }
        }
    }
    for target in bits(targets_changed) {
        roots |= 1u128 << target;
        // 目标攻防数量变化，影响所有以该目标为端点的共同枢纽。
        roots |= before.incoming[target] | after.incoming[target];
    }
    for source in bits(before.constrained ^ after.constrained) {
        roots |= 1u128 << source;
        roots |= (before.controls[source] & before.board.occupied)
            | (after.controls[source] & after.board.occupied);
    }
    (sources, roots)
}
