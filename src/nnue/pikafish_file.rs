//! 官方 Pikafish 2026 `.nnue` 文件的标量读取与量化前向。
//! 文件中的 FC 权重按输出行存储；本实现不做 CPU SIMD 排列。

use std::fs;
use std::path::Path;

use crate::nnue::pikafish;
use crate::xiangqi::Position;

const VERSION: u32 = 0x6a44_8afa;
const MAGIC: &[u8] = b"COMPRESSED_LEB128";
const PSQ: usize = pikafish::PSQ_INPUTS;
const THREAT: usize = pikafish::THREAT_INPUTS;
const WIDTH: usize = pikafish::TRANSFORMER_WIDTH;
const BUCKETS: usize = pikafish::LAYER_STACKS;

#[derive(Debug)]
struct Stack {
    b0: Vec<i32>,
    w0: Vec<u8>,
    b1: Vec<i32>,
    w1: Vec<u8>,
    b2: i32,
    w2: Vec<u8>,
}

/// 文件以官方格式完整解析后才构造；评价值以行棋方视角的棋子分返回。
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
        let stack = &self.stacks[bucket];
        let fc0 = affine(&transformed, &stack.b0, &stack.w0, 32);
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
        Ok(((psqt[0] - psqt[1]) / 2) / 16 + positional / 16)
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
}
