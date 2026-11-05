//! ChessDB DTC/DTM 只读查询：返回亚洲规则的表值，不是当前完整历史的证明。
//! 表文件在对象存活期间必须保持不变；解压缓存上限64MiB，不进入MCTS。
use crate::xiangqi::Position;
use byteorder::{ByteOrder, LittleEndian as LE};
use lzma_rust2::LzmaReader;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char},
    fs,
    io::{self, Read},
    path::Path,
};
unsafe extern "C" {
    fn chineseai_tb_init(data: *const u8, len: usize) -> i32;
    fn chineseai_tb_index(
        fen: *const c_char,
        names: *mut c_char,
        offset: *mut u64,
        side: *mut i32,
    ) -> i32;
}
const MAX_BLOCK_CACHE: usize = 64 * 1024 * 1024;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn bytes(image: &[u8], offset: usize, size: usize) -> io::Result<&[u8]> {
    image
        .get(
            offset
                ..offset
                    .checked_add(size)
                    .ok_or_else(|| invalid("table offset overflow"))?,
        )
        .ok_or_else(|| invalid("table truncated"))
}

struct Side {
    single: Option<u16>,
    flags: u8,
    tail: usize,
    block_size: usize,
    blocks: usize,
    headers: usize,
    data: usize,
}
struct Table {
    image: memmap2::Mmap,
    sides: Vec<Side>,
}
impl Table {
    fn parse(image: memmap2::Mmap) -> io::Result<Self> {
        let count = LE::read_u32(bytes(&image, 4, 4)?) as usize & 3;
        if !(1..=2).contains(&count) {
            return Err(invalid("invalid table side count"));
        }
        let mut offset = 8;
        let mut sides = Vec::new();
        let mut sizes = Vec::new();
        for _ in 0..count {
            let header = bytes(&image, offset, 2)?;
            offset += 2;
            let single = (header[0] & 0x80 != 0).then_some(header[1] as u16);
            let mut side = Side {
                single,
                flags: header[1],
                tail: 0,
                block_size: 0,
                blocks: 0,
                headers: 0,
                data: 0,
            };
            let mut size = 0;
            if single.is_none() {
                let header = bytes(&image, offset, 20)?;
                offset += 20;
                side.tail = LE::read_u32(&header[..4]) as usize;
                side.block_size = LE::read_u32(&header[4..8]) as usize;
                side.blocks = LE::read_u32(&header[8..12]) as usize;
                size = usize::try_from(LE::read_u64(&header[12..20]))
                    .map_err(|_| invalid("table too large"))?;
                if side.block_size < 2
                    || side.block_size % 2 != 0
                    || side.block_size > 1024 * 1024
                    || side.tail == 0
                    || side.tail % 2 != 0
                    || side.tail > side.block_size
                    || side.blocks == 0
                {
                    return Err(invalid("invalid table block layout"));
                }
            }
            sides.push(side);
            sizes.push(size);
        }
        for side in &mut sides {
            if side.single.is_none() {
                side.headers = offset;
                bytes(&image, offset, side.blocks * 8)?;
                offset += side.blocks * 8;
            }
        }
        for (side, size) in sides.iter_mut().zip(sizes) {
            if side.single.is_none() {
                offset = (offset + 63) & !63;
                side.data = offset;
                bytes(&image, offset, size)?;
                offset += size;
            }
        }
        Ok(Self { image, sides })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TablebaseVerdict {
    /// 行棋方视角，来自两种度量一致的胜负和。
    pub wdl: i8,
    /// 原始表距离编码（终局编码为1），只用于证明排序，不作为MLH标签。
    pub dtm_rank: u16,
    pub dtc_steps: u16,
    pub dtc_order: u16,
}

pub struct EndgameTablebases {
    tables: Vec<Table>,
    index: HashMap<(String, bool), usize>,
    cache: HashMap<(usize, usize, usize), Vec<u8>>,
    cache_bytes: usize,
    verdicts: HashMap<String, Option<TablebaseVerdict>>,
    certified_draws: HashMap<String, Vec<crate::xiangqi::Move>>,
    pub image_bytes: usize,
}

impl EndgameTablebases {
    pub fn load(root: &Path) -> io::Result<Self> {
        let group = fs::read(root.join("GROUP.dat"))?;
        if format!("{:x}", Sha256::digest(&group))
            != "bfa5048fa2293ea3da9ebe9e98dcfefd840229add8a303f12a386b3fe1bc2630"
        {
            return Err(invalid("unexpected ChessDB GROUP.dat"));
        }
        // 上游棋子组索引只初始化一次，表对象只在训练线程持有。
        if unsafe { chineseai_tb_init(group.as_ptr(), group.len()) } != 1 {
            return Err(invalid("ChessDB index initialization failed"));
        }
        let mut out = Self {
            tables: Vec::new(),
            index: HashMap::new(),
            cache: HashMap::new(),
            cache_bytes: 0,
            verdicts: HashMap::new(),
            certified_draws: HashMap::new(),
            image_bytes: 0,
        };
        for (folder, extension, dtm) in [("EGTB_DTC", "lzdtc", false), ("EGTB_DTM", "lzdtm", true)]
        {
            for entry in fs::read_dir(root.join(folder))? {
                let path = entry?.path();
                if path.extension().and_then(|s| s.to_str()) != Some(extension) {
                    continue;
                }
                let size = usize::try_from(fs::metadata(&path)?.len())
                    .map_err(|_| invalid("table too large"))?;
                let material = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| invalid("invalid table filename"))?
                    .to_owned();
                // 只读映射；训练期间禁止修改或替换表文件。
                let file = fs::File::open(&path)?;
                let image = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let table = Table::parse(image)?;
                out.index.insert((material, dtm), out.tables.len());
                out.tables.push(table);
                out.image_bytes += size;
            }
        }
        if out.tables.is_empty() {
            return Err(invalid("no ChessDB tables found"));
        }
        Ok(out)
    }

    fn raw(&mut self, table: usize, side: usize, offset: u64) -> io::Result<Option<(u16, u8)>> {
        let file = &self.tables[table];
        let Some(layout) = file.sides.get(side) else {
            return Ok(None);
        };
        if let Some(raw) = layout.single {
            return Ok(Some((raw, 0)));
        }
        let entries = layout.block_size / 2;
        let offset = usize::try_from(offset).map_err(|_| invalid("table index overflow"))?;
        let block = offset / entries;
        if block >= layout.blocks {
            return Err(invalid("table index outside domain"));
        }
        let key = (table, side, block);
        if !self.cache.contains_key(&key) {
            let header = LE::read_u64(bytes(&file.image, layout.headers + block * 8, 8)?);
            let size = (header & 0xfffff) as usize;
            let start = layout.data + (header >> 20) as usize;
            if size < 5 {
                return Err(invalid("invalid LZMA block"));
            }
            let data = bytes(&file.image, start, size)?;
            let props = &data[size - 5..];
            let decoded_size = if block + 1 == layout.blocks {
                layout.tail
            } else {
                layout.block_size
            };
            let dictionary = LE::read_u32(&props[1..]);
            if dictionary > 64 * 1024 * 1024 {
                return Err(invalid("LZMA dictionary exceeds limit"));
            }
            let mut reader = LzmaReader::new_with_props(
                &data[..size - 5],
                decoded_size as u64,
                props[0],
                dictionary,
                None,
            )?;
            let mut decoded = vec![0; decoded_size];
            reader.read_exact(&mut decoded)?;
            if self.cache_bytes + decoded_size > MAX_BLOCK_CACHE {
                self.cache.clear();
                self.cache_bytes = 0;
            }
            self.cache_bytes += decoded_size;
            self.cache.insert(key, decoded);
        }
        let data = &self.cache[&key];
        Ok(Some((
            LE::read_u16(bytes(data, (offset % entries) * 2, 2)?),
            layout.flags,
        )))
    }

    pub fn materials(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .index
            .keys()
            .filter(|(n, m)| *m && self.index.contains_key(&(n.clone(), false)))
            .map(|(n, _)| n.clone())
            .collect();
        names.sort();
        names
    }
    pub fn probe(&mut self, position: &Position) -> io::Result<Option<TablebaseVerdict>> {
        let key = position
            .to_fen()
            .split_whitespace()
            .take(2)
            .collect::<Vec<_>>()
            .join(" ");
        if let Some(&v) = self.verdicts.get(&key) {
            return Ok(v);
        }
        let v = self.probe_uncached(position)?;
        if self.verdicts.len() < 524288 {
            self.verdicts.insert(key, v);
        }
        Ok(v)
    }
    fn probe_uncached(&mut self, position: &Position) -> io::Result<Option<TablebaseVerdict>> {
        if (0..90)
            .filter(|&square| position.piece_at(square).is_some())
            .count()
            > 13
        {
            return Ok(None);
        }
        let fen = CString::new(position.to_fen()).unwrap();
        let mut names = [0 as c_char; 33];
        let mut offset = 0;
        let mut side = 0;
        if unsafe { chineseai_tb_index(fen.as_ptr(), names.as_mut_ptr(), &mut offset, &mut side) }
            == 0
        {
            return Ok(None);
        }
        let material = unsafe { CStr::from_ptr(names.as_ptr()) }
            .to_str()
            .map_err(|_| invalid("invalid material"))?
            .to_owned();
        let (Some(&dtc), Some(&dtm)) = (
            self.index.get(&(material.clone(), false)),
            self.index.get(&(material, true)),
        ) else {
            return Ok(None);
        };
        let Some((dtc, flags)) = self.raw(dtc, side as usize, offset)? else {
            return Ok(None);
        };
        let Some((dtm, _)) = self.raw(dtm, side as usize, offset)? else {
            return Ok(None);
        };
        let dtc_order = (dtc >> 10) + if flags != 0 && dtc & 512 != 0 { 64 } else { 0 };
        let dtc_steps = dtc & if flags != 0 { 0x1ff } else { 0x3ff };
        let dtm_rank = dtm & 0x7ff;
        let dtc_wdl = if dtc_steps == 0 {
            0
        } else if dtc_steps & 1 == 0 {
            1
        } else {
            -1
        };
        let dtm_wdl = if dtm_rank == 0 {
            0
        } else if dtm & 0x800 != 0 {
            1
        } else {
            -1
        };
        if dtc_wdl != dtm_wdl {
            return Ok(None);
        }
        Ok(Some(TablebaseVerdict {
            wdl: dtm_wdl,
            dtm_rank,
            dtc_steps,
            dtc_order,
        }))
    }
}

/// 构造独立新残局监督。表仅引导候选，决定性标签须经当前完整棋规证明。
/// 不能证明的和棋、缺表、证明超时一律返回None，不伪造标签。
pub fn certified_sample(
    tables: &mut EndgameTablebases,
    position: &Position,
    budget: usize,
) -> io::Result<Option<super::AzTrainingSample>> {
    certified_with_history(tables, position, &position.initial_rule_history(), budget)
}
fn certified_with_history(
    tables: &mut EndgameTablebases,
    position: &Position,
    source_history: &[crate::xiangqi::RuleHistoryEntry],
    budget: usize,
) -> io::Result<Option<super::AzTrainingSample>> {
    use crate::xiangqi::RuleOutcome;
    let history = source_history.to_vec();
    if position.rule_outcome_with_history(&history).is_some() {
        return Ok(None);
    }
    let Some(verdict) = tables.probe(position)? else {
        return Ok(None);
    };
    if verdict.wdl == 0 {
        if history.len() != 1
            || position.halfmove_clock() != 0
            || position.rule60_max_ply().is_none()
            || !position.repetition_draw_enabled()
        {
            return Ok(None);
        }
        let Some(plan) = tables.draw_policy(position) else {
            return Ok(None);
        };
        let moves = position.legal_moves_with_rules(&history);
        let count = moves.iter().filter(|m| plan.contains(m)).count();
        if count == 0 {
            return Ok(None);
        }
        let policy = moves
            .iter()
            .map(|m| {
                if plan.contains(m) {
                    1. / count as f32
                } else {
                    0.
                }
            })
            .collect();
        return Ok(Some(supervised_sample(
            position, &history, &moves, policy, 0,
        )));
    }
    if verdict.dtm_rank > 120 {
        return Ok(None);
    }
    let owner = if verdict.wdl > 0 {
        position.side_to_move()
    } else {
        position.side_to_move().opposite()
    };
    if budget == 0 {
        return Ok(None);
    }
    let mut remaining = budget;
    let moves = position.legal_moves_with_rules(&history);
    let mut candidates = Vec::new();
    for (i, &mv) in moves.iter().enumerate() {
        let mut child = position.clone();
        let mut h = history.clone();
        h.push(position.rule_history_entry_after_move(mv));
        child.make_move(mv);
        let terminal = child.rule_outcome_with_history(&h);
        let rank = if terminal == Some(RuleOutcome::Win(owner)) {
            0
        } else {
            tables
                .probe(&child)?
                .filter(|v| v.wdl == if child.side_to_move() == owner { 1 } else { -1 })
                .map_or(u16::MAX, |v| v.dtm_rank)
        };
        candidates.push((rank, i, child, h));
    }
    candidates.sort_by_key(|x| x.0);
    let mut choices = Vec::new();
    for (rank, i, child, mut h) in candidates {
        if verdict.wdl > 0 && rank >= verdict.dtm_rank {
            continue;
        }
        let proven = prove_win(
            tables,
            &child,
            &mut h,
            owner,
            verdict.dtm_rank.saturating_sub(1),
            &mut remaining,
        )?;
        if verdict.wdl < 0 && !proven {
            return Ok(None);
        }
        if proven {
            choices.push((i, rank));
        }
        if verdict.wdl > 0 && proven {
            break;
        }
    }
    if choices.is_empty() {
        return Ok(None);
    }
    let rank = if verdict.wdl > 0 {
        choices.iter().map(|x| x.1).min()
    } else {
        choices.iter().map(|x| x.1).max()
    }
    .unwrap();
    choices.retain(|x| x.1 == rank);
    let mut policy = vec![0.; moves.len()];
    for (i, _) in &choices {
        policy[*i] = 1. / choices.len() as f32;
    }
    Ok(Some(supervised_sample(
        position,
        &history,
        &moves,
        policy,
        verdict.wdl,
    )))
}
fn supervised_sample(
    position: &Position,
    history: &[crate::xiangqi::RuleHistoryEntry],
    moves: &[crate::xiangqi::Move],
    policy: Vec<f32>,
    value: i8,
) -> super::AzTrainingSample {
    use crate::xiangqi::Color;
    let wdl = match value {
        1 => [1., 0., 0.],
        -1 => [0., 0., 1.],
        _ => [0., 1., 0.],
    };
    super::AzTrainingSample {
        features: super::nnue::extract_sparse_features_az(position),
        rule_context: super::rule_context_features(position, &history),
        history_features: super::history_features(position, &history),
        move_indices: moves
            .iter()
            .map(|&m| {
                super::dense_move_index(super::nnue::canonical_move(position.side_to_move(), m))
            })
            .collect(),
        repetition_flags: moves
            .iter()
            .map(|&mv| u8::from(position.move_repeats_history(history, mv)))
            .collect(),
        policy,
        value_wdl: wdl,
        root_search_wdl: wdl,
        value: value as f32,
        side_sign: if position.side_to_move() == Color::Red {
            1.
        } else {
            -1.
        },
        policy_weight: 1.,
        value_weight: 1.,
        moves_left: 0.,
        moves_left_weight: 0.,
        mating_targets: [0.; super::MATING_HEAD_SIZE],
        mating_weight: 0.,
        mating_goal: super::AzMatingGoal::NONE,
        goal_target: 0.,
        goal_weight: 0.,
        search_simulations: 0,
        meta: super::AzSampleMeta::default(),
    }
}
fn prove_win(
    tables: &mut EndgameTablebases,
    position: &Position,
    history: &mut Vec<crate::xiangqi::RuleHistoryEntry>,
    winner: crate::xiangqi::Color,
    depth: u16,
    budget: &mut usize,
) -> io::Result<bool> {
    use crate::xiangqi::RuleOutcome;
    if let Some(outcome) = position.rule_outcome_with_history(history) {
        return Ok(outcome == RuleOutcome::Win(winner));
    }
    if depth == 0 || *budget == 0 {
        return Ok(false);
    }
    *budget -= 1;
    let attack = position.side_to_move() == winner;
    let moves = position.legal_moves_with_rules(history);
    if moves.is_empty() {
        return Ok(position.side_to_move() != winner);
    }
    let mut children = Vec::new();
    for mv in moves {
        let mut child = position.clone();
        child.make_move(mv);
        let rank = tables
            .probe(&child)?
            .filter(|v| {
                v.wdl
                    == if child.side_to_move() == winner {
                        1
                    } else {
                        -1
                    }
            })
            .map_or(u16::MAX, |v| v.dtm_rank);
        children.push((rank, mv, child));
    }
    children.sort_by_key(|c| c.0);
    for (rank, mv, child) in children {
        // 只接受距离严格下降的证明。缺表/距离不下降只允许检查实际终局；
        // 防守应对不能省略。此限制降低覆盖率，不会把未证明分支当作获胜。
        let next_depth = if rank < depth { rank } else { 0 };
        history.push(position.rule_history_entry_after_move(mv));
        let result = prove_win(tables, &child, history, winner, next_depth, budget)?;
        history.pop();
        if attack && result {
            return Ok(true);
        }
        if !attack && !result {
            return Ok(false);
        }
    }
    Ok(!attack)
}
/// 显式普通残局微调：解除仅目标参数训练范围，不改变网络结构或重新初始化参数。
pub fn train_endgame(
    model: &mut super::AzNnue,
    samples: &[super::AzTrainingSample],
    epochs: usize,
    lr: f32,
    batch: usize,
    rng: &mut super::SplitMix64,
) -> Result<super::AzTrainStats, String> {
    model.goal_adaptation_only = false;
    super::train_samples_weighted(
        model,
        samples,
        epochs,
        lr,
        batch,
        rng,
        super::AzTrainLossWeights {
            value: 1.,
            policy: 1.,
            mating: 0.,
            goal: 0.,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_overflow_and_truncated_blocks() {
        assert!(bytes(&[0; 4], usize::MAX, 2).is_err());
        assert!(bytes(&[0; 4], 2, 3).is_err());
        let mut image = memmap2::MmapMut::map_anon(8).unwrap();
        image[4] = 3;
        assert!(Table::parse(image.make_read_only().unwrap()).is_err());
    }
    #[test]
    fn safety_game_does_not_assume_frontiers_or_checking_cycles_drawn() {
        use crate::xiangqi::{Color, Move};
        let p = Position::from_fen("9/4k4/9/9/9/3N5/9/5K3/9/9 w - - 0 1").unwrap();
        let node = |to, quiet| SafetyNode {
            position: p.clone(),
            terminal: None,
            edges: vec![(Move::new(0, 1), to, quiet)],
        };
        assert_eq!(safe_region(&[node(None, true)], Color::Red), [false]);
        assert_eq!(safe_region(&[node(None, true)], Color::Black), [false]);
        assert_eq!(safe_region(&[node(Some(0), false)], Color::Red), [false]);
        assert_eq!(safe_region(&[node(Some(0), true)], Color::Red), [true]);
        let terminal = SafetyNode {
            position: p.clone(),
            terminal: Some(crate::xiangqi::RuleOutcome::Win(Color::Black)),
            edges: vec![],
        };
        assert_eq!(
            safe_region(&[node(Some(1), true), terminal], Color::Red),
            [false, false]
        );
    }
    #[test]
    #[ignore = "需要本地ChessDB表；手动运行 --ignored"]
    fn three_piece_draw_requires_both_safety_certificates() {
        let mut tables = EndgameTablebases::load(Path::new("data/tablebases/chessdb")).unwrap();
        // 黑将可以立即吃马，进入双将和棋。
        let p = Position::from_fen("4k4/4N4/9/9/9/9/9/9/9/5K3 b - - 0 1").unwrap();
        assert_eq!(tables.probe(&p).unwrap().unwrap().wdl, 0);
        let sample = certified_sample(&mut tables, &p, 10000).unwrap().unwrap();
        assert_eq!(sample.value_wdl, [0., 1., 0.]);
        assert_eq!(sample.moves_left_weight, 0.);
        assert!((sample.policy.iter().sum::<f32>() - 1.).abs() < 1e-6);
        let moves = p.legal_moves();
        for (&mv, &prob) in moves.iter().zip(&sample.policy) {
            if prob == 0. {
                continue;
            }
            let mut child = p.clone();
            child.make_move(mv);
            let entry = child.rule_history_entry(Some(p.side_to_move()));
            assert!(!entry.gives_check && entry.chased_mask == 0);
        }
        let win = Position::from_fen("3k5/9/9/9/9/9/9/2R6/9/5K3 w - - 0 1").unwrap();
        assert!(tables.draw_policy(&win).is_none());
    }
    #[test]
    #[ignore = "需要本地ChessDB表；手动运行 --ignored"]
    fn local_wdl_mirror_and_unknown_are_distinct() {
        let mut tables = EndgameTablebases::load(Path::new("data/tablebases/chessdb")).unwrap();
        assert!(tables.materials().contains(&"KRK".to_string()));
        for (fen, wdl) in [
            ("3akab2/9/4b4/9/9/9/9/2R6/9/4K4 w - - 0 1", 0),
            ("3ak4/9/9/9/9/9/9/2R6/9/5K3 w - - 0 1", 1),
            ("9/5k3/9/4r4/9/9/9/3AB4/4K4/9 w - - 0 1", -1),
        ] {
            let p = Position::from_fen(fen).unwrap();
            assert_eq!(tables.probe(&p).unwrap().unwrap().wdl, wdl);
            assert_eq!(tables.probe(&p.mirror_files()).unwrap().unwrap().wdl, wdl);
        }
        assert!(tables.probe(&Position::startpos()).unwrap().is_none());
    }
    #[test]
    #[ignore = "需要本地ChessDB表；手动运行 --ignored"]
    fn completed_teacher_trajectory_has_real_distances() {
        let mut tables = EndgameTablebases::load(Path::new("data/tablebases/chessdb")).unwrap();
        let p = Position::from_fen("3ak4/9/9/9/9/9/9/2R6/9/5K3 w - - 0 1").unwrap();
        let rows = certified_trajectory(&mut tables, &p, 10000)
            .unwrap()
            .unwrap();
        assert!(!rows.is_empty());
        assert_eq!(rows.last().unwrap().moves_left, 1.);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.moves_left, (rows.len() - i) as f32);
            assert_eq!(row.moves_left_weight, 1.);
            assert_eq!(row.value, if row.side_sign > 0. { 1. } else { -1. });
            assert_eq!(row.meta.game_id, p.hash().max(1));
            assert_eq!(row.meta.ply, i as u16);
        }
        assert!(certified_trajectory(&mut tables, &p, 0).unwrap().is_none());
    }
    #[test]
    #[ignore = "需要本地ChessDB表；手动运行 --ignored"]
    fn fortress_rl_draw_reaches_real_terminal_without_teaching_shuffle_policy() {
        use crate::xiangqi::RuleOutcome;
        let mut tables = EndgameTablebases::load(Path::new("data/tablebases/chessdb")).unwrap();
        let mut p = Position::from_fen("3akab2/9/4b4/9/9/9/9/2R6/9/4K4 w - - 0 1").unwrap();
        let rows = draw_rollout(&mut tables, &p).unwrap().unwrap();
        let mut h = p.initial_rule_history();
        for row in rows {
            assert_eq!(row.value_wdl, [0., 1., 0.]);
            assert_eq!(row.policy_weight, 0.);
            assert_eq!(row.moves_left_weight, 0.);
            let moves = p.legal_moves_with_rules(&h);
            let mv = moves[row.policy.iter().position(|&v| v > 0.).unwrap()];
            h.push(p.rule_history_entry_after_move(mv));
            p.make_move(mv);
        }
        assert!(matches!(
            p.rule_outcome_with_history(&h),
            Some(RuleOutcome::Draw(_))
        ));
        assert!(
            draw_rollout(&mut tables, &Position::startpos())
                .unwrap()
                .is_none()
        );
    }
    #[test]
    #[ignore = "需要本地ChessDB表；手动运行 --ignored"]
    fn proof_budget_rule_clock_and_draw_are_not_fabricated_labels() {
        let mut tables = EndgameTablebases::load(Path::new("data/tablebases/chessdb")).unwrap();
        let p = Position::from_fen("3ak4/9/9/9/9/9/9/2R6/9/5K3 w - - 0 1").unwrap();
        assert!(certified_sample(&mut tables, &p, 0).unwrap().is_none());
        let good = certified_sample(&mut tables, &p, 10000).unwrap().unwrap();
        assert_eq!(good.value, 1.);
        assert!((good.policy.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert_eq!(good.moves_left_weight, 0., "证明距离不冒充实际对局长度");
        assert_eq!(good.search_simulations, 0);
        let expired = Position::from_fen("3ak4/9/9/9/9/9/9/2R6/9/5K3 w - - 120 1").unwrap();
        assert!(
            certified_sample(&mut tables, &expired, 10000)
                .unwrap()
                .is_none()
        );
        let draw = Position::from_fen("3akab2/9/4b4/9/9/9/9/2R6/9/4K4 w - - 0 1").unwrap();
        assert!(
            certified_sample(&mut tables, &draw, 10000)
                .unwrap()
                .is_none()
        );
    }
}

/// 生成完整教师收尾轨迹。每一手须经当前历史下的证明，走到真实终局才保留整条轨迹。
/// MLH标签来自这条实际完成的教师对局，不把表距离当成实战长度。
pub fn certified_trajectory(
    tables: &mut EndgameTablebases,
    start: &Position,
    budget: usize,
) -> io::Result<Option<Vec<super::AzTrainingSample>>> {
    use crate::xiangqi::RuleOutcome;
    let mut p = start.clone();
    let mut history = p.initial_rule_history();
    let mut rows = Vec::new();
    let Some(root) = tables.probe(&p)? else {
        return Ok(None);
    };
    if root.wdl == 0 || root.dtm_rank > 120 {
        return Ok(None);
    }
    let winner = if root.wdl > 0 {
        p.side_to_move()
    } else {
        p.side_to_move().opposite()
    };
    for _ in 0..=120 {
        let outcome = p.rule_outcome_with_history(&history).or_else(|| {
            p.legal_moves_with_rules(&history)
                .is_empty()
                .then_some(RuleOutcome::Win(p.side_to_move().opposite()))
        });
        if let Some(outcome) = outcome {
            if outcome != RuleOutcome::Win(winner) || rows.is_empty() {
                return Ok(None);
            }
            let len = rows.len();
            for (i, row) in rows.iter_mut().enumerate() {
                let row: &mut super::AzTrainingSample = row;
                row.moves_left = (len - i) as f32;
                row.moves_left_weight = 1.;
                row.meta.game_id = start.hash().max(1);
                row.meta.ply = i as u16;
            }
            return Ok(Some(rows));
        }
        let Some(row) = certified_with_history(tables, &p, &history, budget)? else {
            return Ok(None);
        };
        if (row.value > 0.) != (p.side_to_move() == winner) {
            return Ok(None);
        }
        let moves = p.legal_moves_with_rules(&history);
        let index = row.policy.iter().position(|&p| p > 0.).unwrap();
        let mv = moves[index];
        rows.push(row);
        history.push(p.rule_history_entry_after_move(mv));
        p.make_move(mv);
    }
    Ok(None)
}

/// 表引导双方保和的完整对局回报，属于离线教师样本，不是当前网络的on-policy采样。
/// 不冒充极小极大和棋证明。
/// 必须每步仍为表和、最终由当前完整棋规实际判和；截断和任一方获胜均丢弃。
/// 只训练WDL，禁止模仿保和循环策略，也不训练和棋的剩余步数。
pub fn draw_rollout(
    tables: &mut EndgameTablebases,
    start: &Position,
) -> io::Result<Option<Vec<super::AzTrainingSample>>> {
    use crate::xiangqi::{PieceKind, RuleOutcome};
    if tables.probe(start)?.is_none_or(|v| v.wdl != 0) {
        return Ok(None);
    }
    let mut p = start.clone();
    let mut history = p.initial_rule_history();
    let mut rows = Vec::new();
    for _ in 0..=240 {
        let terminal = p.rule_outcome_with_history(&history).or_else(|| {
            p.legal_moves_with_rules(&history)
                .is_empty()
                .then_some(RuleOutcome::Win(p.side_to_move().opposite()))
        });
        if let Some(outcome) = terminal {
            return Ok(
                if matches!(outcome, RuleOutcome::Draw(_)) && !rows.is_empty() {
                    Some(rows)
                } else {
                    None
                },
            );
        }
        let moves = p.legal_moves_with_rules(&history);
        let mut best = None;
        for (i, &mv) in moves.iter().enumerate() {
            let mut child = p.clone();
            child.make_move(mv);
            if tables.probe(&child)?.is_none_or(|v| v.wdl != 0) {
                continue;
            }
            let entry = child.rule_history_entry(Some(p.side_to_move()));
            let score = (
                !entry.gives_check,
                entry.chased_mask == 0,
                p.piece_at(mv.from as usize).unwrap().kind == PieceKind::General,
            );
            if best.is_none_or(|(s, _)| score > s) {
                best = Some((score, i));
            }
        }
        let Some((_, i)) = best else {
            return Ok(None);
        };
        let mut policy = vec![0.; moves.len()];
        policy[i] = 1.;
        let mut row = supervised_sample(&p, &history, &moves, policy, 0);
        row.policy_weight = 0.;
        rows.push(row);
        history.push(p.rule_history_entry_after_move(moves[i]));
        p.make_move(moves[i]);
    }
    Ok(None)
}

fn board_key(p: &Position) -> String {
    p.to_fen()
        .split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
}
struct SafetyNode {
    position: Position,
    terminal: Option<crate::xiangqi::RuleOutcome>,
    edges: Vec<(crate::xiangqi::Move, Option<usize>, bool)>,
}
impl EndgameTablebases {
    /// 三子残局的保和策略证书。双方分别验证：己方只用既不将军也不捉子的
    /// 安全走法；对方覆盖所有棋盘合法走法。安全闭包中的循环按当前棋规只能
    /// 判和或使对方犯规，不能归罪于从未将捉的己方。自然限着只会更早判和。
    /// 有限图未展开的边按失败处理，绝不把预算边界当和棋。
    fn draw_policy(&mut self, root: &Position) -> Option<Vec<crate::xiangqi::Move>> {
        use crate::xiangqi::{Color, RuleOutcome};
        let key = board_key(root);
        if let Some(policy) = self.certified_draws.get(&key) {
            return Some(policy.clone());
        }
        if root.occupied_squares().count() > 3 {
            return None;
        }
        let mut nodes = vec![SafetyNode {
            position: root.clone(),
            terminal: None,
            edges: Vec::new(),
        }];
        let mut index = HashMap::from([(key.clone(), 0usize)]);
        let mut cursor = 0;
        while cursor < nodes.len() {
            let mut p = nodes[cursor].position.clone();
            p.set_rule60_max_ply(None);
            let terminal = if !p.has_general(Color::Red) {
                Some(RuleOutcome::Win(Color::Black))
            } else if !p.has_general(Color::Black) {
                Some(RuleOutcome::Win(Color::Red))
            } else {
                p.rule_outcome_with_history(&p.initial_rule_history())
            };
            if terminal.is_some() {
                nodes[cursor].terminal = terminal;
                cursor += 1;
                continue;
            }
            let moves = p.legal_moves();
            if moves.is_empty() {
                nodes[cursor].terminal = Some(RuleOutcome::Win(p.side_to_move().opposite()));
                cursor += 1;
                continue;
            }
            let mut edges = Vec::new();
            for mv in moves {
                let mut child = p.clone();
                child.make_move(mv);
                // after_move条目不保存捉子掩码；此处必须完整重算，不能将其0值当作无捉。
                let entry = child.rule_history_entry(Some(p.side_to_move()));
                let quiet = !entry.gives_check && entry.chased_mask == 0;
                let child_key = board_key(&child);
                let to = if let Some(&id) = index.get(&child_key) {
                    Some(id)
                } else if nodes.len() < 20000 {
                    let id = nodes.len();
                    index.insert(child_key, id);
                    nodes.push(SafetyNode {
                        position: child,
                        terminal: None,
                        edges: Vec::new(),
                    });
                    Some(id)
                } else {
                    None
                };
                edges.push((mv, to, quiet));
            }
            nodes[cursor].edges = edges;
            cursor += 1;
        }
        let red = safe_region(&nodes, Color::Red);
        let black = safe_region(&nodes, Color::Black);
        for (id, node) in nodes.iter().enumerate() {
            if !red[id] || !black[id] || node.terminal.is_some() {
                continue;
            }
            let safe = if node.position.side_to_move() == Color::Red {
                &red
            } else {
                &black
            };
            let plan: Vec<_> = node
                .edges
                .iter()
                .filter(|(_, to, quiet)| *quiet && to.is_some_and(|i| safe[i]))
                .map(|(mv, _, _)| *mv)
                .collect();
            if !plan.is_empty() {
                self.certified_draws.insert(board_key(&node.position), plan);
            }
        }
        self.certified_draws.get(&key).cloned()
    }
}
fn safe_region(nodes: &[SafetyNode], owner: crate::xiangqi::Color) -> Vec<bool> {
    use crate::xiangqi::RuleOutcome;
    use std::collections::VecDeque;
    let mut bad = vec![false; nodes.len()];
    let mut live = vec![0; nodes.len()];
    let mut predecessors = vec![Vec::new(); nodes.len()];
    for (id, node) in nodes.iter().enumerate() {
        if let Some(outcome) = node.terminal {
            bad[id] = matches!(outcome,RuleOutcome::Win(c) if c!=owner);
            continue;
        }
        let own = node.position.side_to_move() == owner;
        for &(_, to, quiet) in &node.edges {
            if own && !quiet {
                continue;
            }
            if let Some(child) = to {
                live[id] += 1;
                predecessors[child].push(id);
            } else if !own {
                bad[id] = true;
            }
        }
        if own && live[id] == 0 {
            bad[id] = true;
        }
    }
    let mut queue: VecDeque<_> = bad
        .iter()
        .enumerate()
        .filter(|(_, b)| **b)
        .map(|(i, _)| i)
        .collect();
    while let Some(child) = queue.pop_front() {
        for &parent in &predecessors[child] {
            if bad[parent] {
                continue;
            }
            let own = nodes[parent].position.side_to_move() == owner;
            if own {
                live[parent] -= 1;
            }
            if !own || live[parent] == 0 {
                bad[parent] = true;
                queue.push_back(parent);
            }
        }
    }
    bad.into_iter().map(|b| !b).collect()
}
