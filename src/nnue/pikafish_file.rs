//! 官方 Pikafish 2026 `.nnue` 文件的标量读取与量化前向。
//! 文件中的 FC 权重按输出行存储；本实现不做 CPU SIMD 排列。

use std::fs;
use std::path::Path;

use crate::nnue::pikafish;
use crate::xiangqi::{Color, PieceKind, Position};

const VERSION: u32 = 0x6a44_8afa;
const MAGIC: &[u8] = b"COMPRESSED_LEB128";
const PSQ: usize = pikafish::PSQ_INPUTS;
const THREAT: usize = pikafish::THREAT_INPUTS;
const WIDTH: usize = pikafish::TRANSFORMER_WIDTH;
const BUCKETS: usize = pikafish::LAYER_STACKS;

/// 将搜索使用的有界 q 分数还原为 Pikafish UCI `score cp` 的内部单位。
/// Pikafish 的搜索信息直接输出内部 Value；`eval` 追踪里的换算值另有定义。
pub fn internal_units_from_q(q: f32) -> i32 {
    ((q.clamp(-0.95, 0.95).atanh() * 600.0).round()) as i32
}

// Pikafish types.h, identical in local release 4c17cee and later b562d6a.
const PIECE_VALUES: [i32; 7] = [0, 219, 187, 720, 1305, 773, 144];

fn material_values(position: &Position) -> (i32, i32, i32) {
    let mut red = 0;
    let mut black = 0;
    let mut wdl_material = 0;
    for sq in 0..90 {
        if let Some(piece) = position.piece_at(sq) {
            let (value, wdl) = match piece.kind {
                PieceKind::General => (PIECE_VALUES[0], 0),
                PieceKind::Advisor => (PIECE_VALUES[1], 2),
                PieceKind::Elephant => (PIECE_VALUES[2], 3),
                PieceKind::Horse => (PIECE_VALUES[3], 5),
                PieceKind::Rook => (PIECE_VALUES[4], 10),
                PieceKind::Cannon => (PIECE_VALUES[5], 5),
                PieceKind::Soldier => (PIECE_VALUES[6], 1),
            };
            match piece.color {
                Color::Red => red += value,
                Color::Black => black += value,
            }
            wdl_material += wdl;
        }
    }
    (red, black, wdl_material)
}

/// 当前 master `scale_evaluation`：返回行棋方视角的内部 Value。
/// `rule60_count` 取自搜索历史；只用 FEN 时可传 `position.halfmove_clock()`。
pub fn scale_evaluation_latest(
    nnue: i32,
    optimism: i32,
    position: &Position,
    rule60_count: u16,
) -> i32 {
    let (red, black, _) = material_values(position);
    let se = if position.side_to_move() == Color::Red {
        red - black
    } else {
        black - red
    };
    let se_norm = se * 1024 / (se.abs() + 1024);
    let nnue_norm = nnue * 1024 / (nnue.abs() + 1024);
    let alignment = se_norm * nnue_norm / 512;
    let base_eval = nnue + nnue * alignment / 65_536 + optimism * alignment / 16_384;
    let mut v = (base_eval as i64 * (80_030 + red + black) as i64 / 80_030) as i32;
    v -= v * i32::from(rule60_count) / 244;
    v.clamp(-31_753, 31_753)
}

/// 本地 2026-09-06 发布版使用的静态评估缩放。对应官方 commit `4c17cee`。
pub fn scale_evaluation_release(
    psqt: i32,
    positional: i32,
    optimism: i32,
    position: &Position,
    rule60_count: u16,
) -> i32 {
    let complexity = (psqt - positional).abs();
    let optimism = optimism + (i64::from(optimism) * i64::from(complexity) / 467) as i32;
    let nnue = psqt + positional;
    let nnue = nnue - (i64::from(nnue) * i64::from(complexity) / 11_698) as i32;
    let mut major_material = 0;
    for sq in 0..90 {
        if let Some(piece) = position.piece_at(sq) {
            if matches!(
                piece.kind,
                PieceKind::Rook | PieceKind::Cannon | PieceKind::Horse
            ) {
                major_material += match piece.kind {
                    PieceKind::Rook => 1305,
                    PieceKind::Cannon => 773,
                    _ => 720,
                };
            }
        }
    }
    let mut v = nnue
        + ((i64::from(nnue) * i64::from(major_material) + i64::from(optimism) * 13_371) / 36_220)
            as i32;
    v -= v * i32::from(rule60_count) / 244;
    v.clamp(-31_753, 31_753)
}

/// 公开源码 `b562d6a` 的 `UCIEngine::to_cp`，及 9/6 发布版的 `eval` 换算。
/// 本地自报 2026-09-25 的二进制 `eval` 输出与此源码不符，不能用于其换算对照。
/// 搜索 `score cp` 直接输出内部 Value，不使用这个换算。
pub fn normalized_cp(value: i32, position: &Position) -> i32 {
    let (_, _, material) = material_values(position);
    let m = material.clamp(17, 110) as f64 / 65.0;
    let a = ((220.598_913_65 * m - 810.357_304_30) * m + 928.681_851_98) * m + 79.839_554_23;
    (100.0 * f64::from(value) / a).round() as i32
}

#[derive(Debug)]
struct Stack {
    b0: Vec<i32>,
    w0: Vec<u8>,
    b1: Vec<i32>,
    w1: Vec<u8>,
    b2: i32,
    w2: Vec<u8>,
}

/// 文件以官方格式完整解析后才构造；评价值以行棋方视角的内部 Value 单位返回。
#[derive(Debug)]
pub struct PikafishNet {
    pub description: String,
    ft_bias: Vec<i16>,
    threat_weights: Vec<u8>,
    threat_psqt: Vec<i32>,
    psq_weights: Vec<u8>,
    psq_psqt: Vec<i32>,
    stacks: Vec<Stack>,
}

/// 搜索中复用的量化特征累加器。按上一评估局面与新局面的活跃特征集合做差，
/// 因而 make/unmake 后仍可直接评估任意祖先或兄弟节点。
#[derive(Debug)]
pub struct PikafishEvalCache {
    psq: [Vec<usize>; 2],
    threats: [Vec<usize>; 2],
    psq_next: [Vec<usize>; 2],
    threats_next: [Vec<usize>; 2],
    accumulators: [Vec<i16>; 2],
    initialized: bool,
}

impl Default for PikafishEvalCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PikafishEvalCache {
    pub fn new() -> Self {
        Self {
            psq: [Vec::new(), Vec::new()],
            threats: [Vec::new(), Vec::new()],
            psq_next: [Vec::new(), Vec::new()],
            threats_next: [Vec::new(), Vec::new()],
            accumulators: [vec![0; WIDTH], vec![0; WIDTH]],
            initialized: false,
        }
    }

    pub fn evaluate_split(
        &mut self,
        net: &PikafishNet,
        position: &Position,
    ) -> Result<(i32, i32), String> {
        let side = position.side_to_move();
        let bucket = pikafish::layer_stack_bucket(position);
        let mut transformed = [0_u8; WIDTH];
        let mut psqt = [0_i32; 2];
        let (red_threats, black_threats) = self.threats_next.split_at_mut(1);
        super::full_threats::fill_threat_features_both(
            position,
            &mut red_threats[0],
            &mut black_threats[0],
        )
        .ok_or("invalid FullThreats position")?;
        // 缓存槽位固定为红、黑视角；输出顺序仍由行棋方决定。
        for (output_idx, perspective) in [side, side.opposite()].into_iter().enumerate() {
            let slot = if perspective == Color::Red { 0 } else { 1 };
            let psq = &mut self.psq_next[slot];
            let threats = &mut self.threats_next[slot];
            psq.clear();
            pikafish::fill_psq_features(position, perspective, psq)
                .ok_or("invalid HalfKAv2_hm position")?;
            psq.sort_unstable();
            threats.sort_unstable();
            let acc = &mut self.accumulators[slot];
            if !self.initialized {
                acc.copy_from_slice(&net.ft_bias);
            }
            update_features(acc, &mut self.psq[slot], psq, &net.psq_weights);
            update_features(acc, &mut self.threats[slot], threats, &net.threat_weights);
            for &index in &self.psq[slot] {
                psqt[output_idx] += net.psq_psqt[index * BUCKETS + bucket];
            }
            for &index in &self.threats[slot] {
                psqt[output_idx] += net.threat_psqt[index * BUCKETS + bucket];
            }
            for i in 0..WIDTH / 2 {
                let a = acc[i].clamp(0, 255) as u32;
                let b = acc[i + WIDTH / 2].clamp(0, 255) as u32;
                transformed[output_idx * WIDTH / 2 + i] = (a * b / 512) as u8;
            }
        }
        self.initialized = true;
        Ok(net.forward_transformed(&transformed, bucket, psqt))
    }

    pub fn evaluate_scaled(
        &mut self,
        net: &PikafishNet,
        position: &Position,
        rule60_count: u16,
    ) -> Result<i32, String> {
        let (psqt, positional) = self.evaluate_split(net, position)?;
        Ok(scale_evaluation_latest(
            psqt + positional,
            0,
            position,
            rule60_count,
        ))
    }
}

fn update_features(acc: &mut [i16], old: &mut Vec<usize>, new: &mut Vec<usize>, weights: &[u8]) {
    let (mut i, mut j) = (0, 0);
    while i < old.len() && j < new.len() {
        if old[i] < new[j] {
            subtract_feature(acc, weights, old[i]);
            i += 1;
        } else if old[i] > new[j] {
            add_feature(acc, weights, new[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    for &index in &old[i..] {
        subtract_feature(acc, weights, index);
    }
    for &index in &new[j..] {
        add_feature(acc, weights, index);
    }
    std::mem::swap(old, new);
}

fn subtract_feature(acc: &mut [i16], weights: &[u8], index: usize) {
    for (value, weight) in acc
        .iter_mut()
        .zip(&weights[index * WIDTH..(index + 1) * WIDTH])
    {
        *value = value.wrapping_sub((*weight as i8) as i16);
    }
}

impl PikafishNet {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let data = if bytes.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
            zstd::decode_all(bytes.as_slice()).map_err(|e| format!("zstd: {e}"))?
        } else {
            bytes
        };
        let mut input = Input::new(&data);
        input.expect_u32(VERSION, "version")?;
        input.expect_u32(network_hash(), "network hash")?;
        let description_len = input.u32()? as usize;
        let description = String::from_utf8_lossy(input.take(description_len)?).into_owned();
        input.expect_u32(transformer_hash(), "transformer hash")?;
        let ft_bias = input.leb_i16(WIDTH)?;
        let threat_weights = input.take(THREAT * WIDTH)?.to_vec();
        let threat_psqt = input.leb_i32(THREAT * BUCKETS)?;
        let psq_weights = input.take(PSQ * WIDTH)?.to_vec();
        let psq_psqt = input.leb_i32(PSQ * BUCKETS)?;
        let mut stacks = Vec::with_capacity(BUCKETS);
        for _ in 0..BUCKETS {
            input.expect_u32(stack_hash(), "layer stack hash")?;
            stacks.push(Stack {
                b0: input.i32s(32)?,
                w0: input.take(32 * WIDTH)?.to_vec(),
                b1: input.i32s(32)?,
                w1: input.take(32 * 64)?.to_vec(),
                b2: input.i32()?,
                w2: input.take(128)?.to_vec(),
            });
        }
        if input.remaining() != 0 {
            return Err(format!("trailing NNUE bytes: {}", input.remaining()));
        }
        Ok(Self {
            description,
            ft_bias,
            threat_weights,
            threat_psqt,
            psq_weights,
            psq_psqt,
            stacks,
        })
    }

    pub fn evaluate(&self, position: &Position) -> Result<i32, String> {
        self.evaluate_split(position)
            .map(|(psqt, positional)| psqt + positional)
    }

    /// 发布版评估需要分开的 PSQT 和 positional 分量。
    pub fn evaluate_split(&self, position: &Position) -> Result<(i32, i32), String> {
        let side = position.side_to_move();
        let bucket = pikafish::layer_stack_bucket(position);
        let mut transformed = [0_u8; WIDTH];
        let mut psqt = [0_i32; 2];
        for (perspective_idx, perspective) in [side, side.opposite()].into_iter().enumerate() {
            let mut psq = Vec::new();
            let mut threats = Vec::new();
            pikafish::fill_psq_features(position, perspective, &mut psq)
                .ok_or("invalid HalfKAv2_hm position")?;
            pikafish::fill_threat_features(position, perspective, &mut threats)
                .ok_or("invalid FullThreats position")?;
            let mut acc = self.ft_bias.clone();
            for index in psq {
                add_feature(&mut acc, &self.psq_weights, index);
                psqt[perspective_idx] += self.psq_psqt[index * BUCKETS + bucket];
            }
            for index in threats {
                add_feature(&mut acc, &self.threat_weights, index);
                psqt[perspective_idx] += self.threat_psqt[index * BUCKETS + bucket];
            }
            for i in 0..WIDTH / 2 {
                let a = acc[i].clamp(0, 255) as u32;
                let b = acc[i + WIDTH / 2].clamp(0, 255) as u32;
                transformed[perspective_idx * WIDTH / 2 + i] = (a * b / 512) as u8;
            }
        }
        Ok(self.forward_transformed(&transformed, bucket, psqt))
    }

    fn forward_transformed(
        &self,
        transformed: &[u8; WIDTH],
        bucket: usize,
        psqt: [i32; 2],
    ) -> (i32, i32) {
        let stack = &self.stacks[bucket];
        let fc0 = affine(transformed, &stack.b0, &stack.w0, 32);
        let mut ac0 = [0_u8; 64];
        activation_pair(&fc0, &mut ac0, 7);
        let fc1 = affine(&ac0, &stack.b1, &stack.w1, 32);
        let mut ac1 = [0_u8; 64];
        activation_pair(&fc1, &mut ac1, 6);
        let mut ac2 = [0_u8; 128];
        ac2[..64].copy_from_slice(&ac0);
        ac2[64..].copy_from_slice(&ac1);
        let positional =
            affine(&ac2, &[stack.b2], &stack.w2, 1)[0].wrapping_add(fc0[30].wrapping_sub(fc0[31]));
        let positional = (positional as i64 * (600 * 16) / (128 * 64 * 2)) as i32;
        (((psqt[0] - psqt[1]) / 2) / 16, positional / 16)
    }

    /// 当前官方源码的静态评估，返回行棋方内部 Value。
    pub fn evaluate_scaled(&self, position: &Position, rule60_count: u16) -> Result<i32, String> {
        self.evaluate(position)
            .map(|nnue| scale_evaluation_latest(nnue, 0, position, rule60_count))
    }

    /// 2026-09-06 发布版的缩放算法，仅用于旧版对照。
    pub fn evaluate_scaled_release(
        &self,
        position: &Position,
        rule60_count: u16,
    ) -> Result<i32, String> {
        self.evaluate_split(position).map(|(psqt, positional)| {
            scale_evaluation_release(psqt, positional, 0, position, rule60_count)
        })
    }
}

fn add_feature(acc: &mut [i16], weights: &[u8], index: usize) {
    for (value, weight) in acc
        .iter_mut()
        .zip(&weights[index * WIDTH..(index + 1) * WIDTH])
    {
        *value = value.wrapping_add((*weight as i8) as i16);
    }
}

fn affine(input: &[u8], biases: &[i32], weights: &[u8], outputs: usize) -> Vec<i32> {
    (0..outputs)
        .map(|row| {
            let mut sum = biases[row];
            for (&x, &w) in input
                .iter()
                .zip(&weights[row * input.len()..][..input.len()])
            {
                sum = sum.wrapping_add(x as i32 * (w as i8) as i32);
            }
            sum
        })
        .collect()
}

fn activation_pair(input: &[i32], output: &mut [u8], shift: u32) {
    for (i, &x) in input.iter().enumerate() {
        output[i] = ((x as i64 * x as i64) >> (2 * shift + 7)).min(127) as u8;
        output[i + input.len()] = (x >> shift).clamp(0, 127) as u8;
    }
}

fn transformer_hash() -> u32 {
    0x2e6b_9d04_u32.rotate_left(1) ^ 0x7f23_4cb8 ^ (WIDTH as u32 * 2)
}

fn stack_hash() -> u32 {
    let mut h = 0xec42_e90d ^ (WIDTH as u32 * 2);
    for outputs in [32, 32, 1] {
        h = 0xcc03_dae4_u32.wrapping_add(outputs) ^ h.rotate_right(1);
        if outputs != 1 {
            h = 0x538d_24c7_u32.wrapping_add(h);
        }
    }
    h
}

fn network_hash() -> u32 {
    transformer_hash() ^ stack_hash()
}

struct Input<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Input<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
    fn remaining(&self) -> usize {
        self.data.len() - self.offset
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self.offset.checked_add(len).ok_or("NNUE length overflow")?;
        let bytes = self.data.get(self.offset..end).ok_or("truncated NNUE")?;
        self.offset = end;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(self.u32()? as i32)
    }
    fn i32s(&mut self, len: usize) -> Result<Vec<i32>, String> {
        (0..len).map(|_| self.i32()).collect()
    }
    fn expect_u32(&mut self, expected: u32, label: &str) -> Result<(), String> {
        let got = self.u32()?;
        if got == expected {
            Ok(())
        } else {
            Err(format!(
                "{label}: expected {expected:#010x}, got {got:#010x}"
            ))
        }
    }
    fn leb_i16(&mut self, len: usize) -> Result<Vec<i16>, String> {
        self.leb(len)?
            .into_iter()
            .map(|v| i16::try_from(v).map_err(|_| "i16 LEB128 overflow".to_owned()))
            .collect()
    }
    fn leb_i32(&mut self, len: usize) -> Result<Vec<i32>, String> {
        self.leb(len)
    }
    fn leb(&mut self, len: usize) -> Result<Vec<i32>, String> {
        if self.take(MAGIC.len())? != MAGIC {
            return Err("missing COMPRESSED_LEB128 header".into());
        }
        let bytes = self.u32()? as usize;
        let compressed = self.take(bytes)?;
        let mut offset = 0;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            let mut value = 0_u32;
            let mut shift = 0;
            let last = loop {
                let byte = *compressed.get(offset).ok_or("truncated LEB128 block")?;
                offset += 1;
                value |= ((byte & 0x7f) as u32).wrapping_shl(shift);
                shift += 7;
                if byte & 0x80 == 0 {
                    break byte;
                }
                if shift > 35 {
                    return Err("LEB128 integer too long".into());
                }
            };
            if shift < 32 && last & 0x40 != 0 {
                value |= !((1_u32 << shift) - 1);
            }
            values.push(value as i32);
        }
        if offset != compressed.len() {
            return Err("LEB128 block length mismatch".into());
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure_hashes_match_official_net() {
        assert_eq!(transformer_hash(), 0x23f4_7eb0);
        assert_eq!(stack_hash(), 0x6333_7116);
        assert_eq!(network_hash(), 0x40c7_0fa6);
    }

    #[test]
    fn search_q_returns_internal_score_units() {
        for raw in [-600_i32, -97, 0, 97, 600] {
            let q = (raw as f32 / 600.0).tanh();
            assert_eq!(internal_units_from_q(q), raw);
        }
    }

    #[test]
    fn malformed_leb_block_is_rejected() {
        let mut data = Vec::from(MAGIC);
        data.extend_from_slice(&1_u32.to_le_bytes());
        data.push(0x80);
        assert!(Input::new(&data).leb_i16(1).is_err());
    }

    #[test]
    fn official_file_three_positions() {
        let path = Path::new("tools/pikafish.nnue");
        if !path.exists() {
            return;
        }
        let net = PikafishNet::load(path).unwrap();
        let mut position = Position::startpos();
        for (move_uci, expected) in [(None, 97), (Some("h2e2"), -95), (Some("h7e7"), 129)] {
            if let Some(mv) = move_uci {
                let parsed = position.parse_uci_move(mv).unwrap();
                position.make_move(parsed);
            }
            let actual = net.evaluate(&position).unwrap();
            println!("{move_uci:?}: {actual} expected {expected}");
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn september_6_trace_final_evaluation() {
        let path = Path::new("tools/pikafish.nnue");
        if !path.exists() {
            return;
        }
        let net = PikafishNet::load(path).unwrap();
        let mut position = Position::startpos();
        let initial = net.evaluate_scaled_release(&position, 0).unwrap();
        assert_eq!(initial, 126);
        assert_eq!(normalized_cp(initial, &position), 32);
        assert_eq!(
            normalized_cp(
                net.evaluate_scaled_release(&position, 60).unwrap(),
                &position
            ),
            24
        );
        position.make_move(position.parse_uci_move("h2e2").unwrap());
        let child = net.evaluate_scaled_release(&position, 1).unwrap();
        assert_eq!(normalized_cp(-child, &position), 31);
    }

    #[test]
    fn september_22_source_scale() {
        let path = Path::new("tools/pikafish.nnue");
        if !path.exists() {
            return;
        }
        let net = PikafishNet::load(path).unwrap();
        let position = Position::startpos();
        assert_eq!(net.evaluate(&position).unwrap(), 97);
        assert_eq!(net.evaluate_scaled(&position, 0).unwrap(), 114);
        assert_eq!(net.evaluate_scaled(&position, 60).unwrap(), 86);
    }

    #[test]
    fn incremental_matches_full_across_legal_make_unmake() {
        let path = Path::new("tools/pikafish.nnue");
        if !path.exists() {
            return;
        }
        let net = PikafishNet::load(path).unwrap();
        let mut cache = PikafishEvalCache::new();
        let mut position = Position::startpos();
        let mut history = Vec::new();
        let mut state = 0x9e37_79b9_u64;
        for _ in 0..80 {
            assert_eq!(
                cache.evaluate_split(&net, &position).unwrap(),
                net.evaluate_split(&position).unwrap()
            );
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mv = moves[(state as usize) % moves.len()];
            let undo = position.make_move(mv);
            history.push((mv, undo));
        }
        while let Some((mv, undo)) = history.pop() {
            position.unmake_move(mv, undo);
            assert_eq!(
                cache.evaluate_split(&net, &position).unwrap(),
                net.evaluate_split(&position).unwrap()
            );
        }
    }

    #[test]
    #[ignore = "manual NNUE phase timing"]
    fn profile_quantized_evaluation_phases() {
        let path = Path::new("tools/pikafish.nnue");
        if !path.exists() {
            return;
        }
        let net = PikafishNet::load(path).unwrap();
        let mut position = Position::startpos();
        let mut positions = Vec::new();
        for ply in 0..80 {
            positions.push(position.clone());
            let legal = position.legal_moves();
            if legal.is_empty() {
                break;
            }
            position.make_move(legal[(ply * 37 + 11) % legal.len()]);
        }
        let mut red = Vec::new();
        let mut black = Vec::new();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            for position in &positions {
                super::super::full_threats::fill_threat_features_both(
                    position, &mut red, &mut black,
                )
                .unwrap();
                std::hint::black_box((&red, &black));
            }
        }
        let threats = start.elapsed();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            for position in &positions {
                pikafish::fill_psq_features(position, Color::Red, &mut red).unwrap();
                pikafish::fill_psq_features(position, Color::Black, &mut black).unwrap();
                std::hint::black_box((&red, &black));
            }
        }
        let psq = start.elapsed();
        let mut cache = PikafishEvalCache::new();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            for position in &positions {
                std::hint::black_box(cache.evaluate_split(&net, position).unwrap());
            }
        }
        let cached = start.elapsed();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            for position in &positions {
                std::hint::black_box(net.evaluate_split(position).unwrap());
            }
        }
        let full = start.elapsed();
        eprintln!(
            "NNUE 20x{} positions: threat={threats:?} psq={psq:?} cached={cached:?} full={full:?}",
            positions.len()
        );
    }
}
