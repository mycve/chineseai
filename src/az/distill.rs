use std::{error::Error, fs::File, io::Read, path::Path};

use ndarray::{Array2, ArrayD, Axis, Ix1, Ix2, Ix4};
use ndarray_npy::NpzReader;
use zip::ZipArchive;

use crate::{
    nnue::{canonical_move, extract_sparse_features_v4_canonical},
    xiangqi::{BOARD_FILES, BOARD_RANKS, Position},
};

use super::{
    AzTrainingSample, DENSE_MOVE_SPACE, dense_move_index, dense_move_to_move, extract_board_planes,
    extract_value_relation_features,
};

const DISTILL_PLANES_PER_FRAME: usize = 14;
const DISTILL_FRAMES: usize = 9;
const DISTILL_OBS_PLANES: usize = DISTILL_PLANES_PER_FRAME * DISTILL_FRAMES;
const DISTILL_TOP_K: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct AzDistillLoadOptions {
    pub max_rows: Option<usize>,
    pub validate_legal: bool,
}

impl Default for AzDistillLoadOptions {
    fn default() -> Self {
        Self {
            max_rows: None,
            validate_legal: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzDistillLoadStats {
    pub rows: usize,
    pub samples: usize,
    pub skipped_positions: usize,
    pub policy_entries: usize,
    pub policy_renormalized: usize,
    pub legal_checked: usize,
    pub legal_exact: usize,
    pub legal_overlap_sum: usize,
    pub legal_union_sum: usize,
}

pub fn load_distill_npz_samples(
    path: &Path,
    options: AzDistillLoadOptions,
) -> Result<(Vec<AzTrainingSample>, AzDistillLoadStats), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut npz = NpzReader::new(file)?;
    let obs: ArrayD<u8> = npz.by_name("obs.npy")?;
    let obs = obs.into_dimensionality::<Ix4>()?;
    let policy_idx: ArrayD<u16> = npz.by_name("policy_idx.npy")?;
    let policy_idx = policy_idx.into_dimensionality::<Ix2>()?;
    let policy_prob = read_f16_npz_2d(path, "policy_prob.npy")?;
    let value_tgt: ArrayD<f32> = npz.by_name("value_tgt.npy")?;
    let value_tgt = value_tgt.into_dimensionality::<Ix1>()?;
    let legal_mask = if options.validate_legal {
        let legal_mask: ArrayD<u8> = npz.by_name("legal_mask.npy")?;
        Some(legal_mask.into_dimensionality::<Ix2>()?)
    } else {
        None
    };

    let rows = obs.len_of(Axis(0));
    if obs.shape()[1..] != [DISTILL_OBS_PLANES, BOARD_RANKS, BOARD_FILES] {
        return Err(format!(
            "unexpected obs shape {:?}; expected (N,{DISTILL_OBS_PLANES},{BOARD_RANKS},{BOARD_FILES})",
            obs.shape()
        )
        .into());
    }
    if policy_idx.shape() != [rows, DISTILL_TOP_K] || policy_prob.shape() != [rows, DISTILL_TOP_K] {
        return Err(format!(
            "unexpected policy shapes idx={:?} prob={:?}; expected (N,{DISTILL_TOP_K})",
            policy_idx.shape(),
            policy_prob.shape()
        )
        .into());
    }
    if value_tgt.len() != rows {
        return Err(format!(
            "unexpected value_tgt len {}; expected {rows}",
            value_tgt.len()
        )
        .into());
    }

    let limit = options.max_rows.map_or(rows, |max_rows| max_rows.min(rows));
    let mut samples = Vec::with_capacity(limit);
    let mut stats = AzDistillLoadStats {
        rows: limit,
        ..Default::default()
    };

    for row in 0..limit {
        let Some(fen) = obs_row_to_fen(&obs, row) else {
            stats.skipped_positions += 1;
            continue;
        };
        let Ok(position) = Position::from_fen(&fen) else {
            stats.skipped_positions += 1;
            continue;
        };

        if let Some(mask) = &legal_mask {
            let observed = unpack_legal_mask_row(mask, row);
            let expected = legal_dense_mask(&position);
            let overlap = observed
                .iter()
                .zip(expected.iter())
                .filter(|(left, right)| **left && **right)
                .count();
            let union = observed
                .iter()
                .zip(expected.iter())
                .filter(|(left, right)| **left || **right)
                .count();
            if overlap == union {
                stats.legal_exact += 1;
            }
            stats.legal_checked += 1;
            stats.legal_overlap_sum += overlap;
            stats.legal_union_sum += union;
        }

        let side = position.side_to_move();
        let legal_moves_actual = legal_dense_moves(&position);
        let legal_moves = legal_moves_actual
            .iter()
            .filter_map(|&move_index| dense_move_to_move(move_index))
            .map(|mv| dense_move_index(canonical_move(side, mv)))
            .collect::<Vec<_>>();
        let mut dense_targets = vec![0.0f32; DENSE_MOVE_SPACE];
        let mut prob_sum = 0.0f32;
        for slot in 0..DISTILL_TOP_K {
            let move_index = policy_idx[[row, slot]] as usize;
            let prob = policy_prob[[row, slot]].max(0.0);
            if move_index >= DENSE_MOVE_SPACE || prob <= 0.0 {
                continue;
            }
            let Some(mv) = dense_move_to_move(move_index) else {
                continue;
            };
            let canonical_index = dense_move_index(canonical_move(side, mv));
            dense_targets[canonical_index] += prob;
            prob_sum += prob;
        }
        if legal_moves.is_empty() || prob_sum <= 0.0 {
            stats.skipped_positions += 1;
            continue;
        }
        if (prob_sum - 1.0).abs() > 1e-3 {
            stats.policy_renormalized += 1;
        }
        let policy = legal_moves
            .iter()
            .map(|&move_index| dense_targets[move_index] / prob_sum)
            .collect::<Vec<_>>();
        let legal_moves_raw = legal_moves_actual
            .iter()
            .filter_map(|&move_index| dense_move_to_move(move_index))
            .collect::<Vec<_>>();
        let mut value_relation = Vec::new();
        extract_value_relation_features(&position, &legal_moves_raw, &mut value_relation);

        let mut board = Vec::new();
        extract_board_planes(&position, &[], &mut board);
        samples.push(AzTrainingSample {
            features: extract_sparse_features_v4_canonical(&position, &[]),
            board,
            move_indices: legal_moves,
            policy,
            value_relation,
            value: value_tgt[row].clamp(-1.0, 1.0),
            side_sign: -1.0,
        });
        stats.samples += 1;
        stats.policy_entries += samples.last().map_or(0, |sample| sample.policy.len());
    }

    Ok((samples, stats))
}

fn obs_row_to_fen(obs: &ndarray::Array<u8, Ix4>, row: usize) -> Option<String> {
    let mut fen = String::new();
    for rank in 0..BOARD_RANKS {
        if rank > 0 {
            fen.push('/');
        }
        let mut empty = 0usize;
        for file in 0..BOARD_FILES {
            let piece = obs_square_to_fen_piece(obs, row, rank, file)?;
            if let Some(ch) = piece {
                if empty > 0 {
                    fen.push(char::from_digit(empty as u32, 10)?);
                    empty = 0;
                }
                fen.push(ch);
            } else {
                empty += 1;
            }
        }
        if empty > 0 {
            fen.push(char::from_digit(empty as u32, 10)?);
        }
    }
    fen.push_str(" b");
    Some(fen)
}

fn obs_square_to_fen_piece(
    obs: &ndarray::Array<u8, Ix4>,
    row: usize,
    rank: usize,
    file: usize,
) -> Option<Option<char>> {
    let mut out = None;
    for plane in 0..DISTILL_PLANES_PER_FRAME {
        if obs[[row, plane, rank, file]] == 0 {
            continue;
        }
        if out.is_some() {
            return None;
        }
        out = distill_plane_to_fen_piece(plane);
    }
    Some(out)
}

fn distill_plane_to_fen_piece(plane: usize) -> Option<char> {
    match plane {
        0 => Some('k'),
        1 => Some('a'),
        2 => Some('b'),
        3 => Some('n'),
        4 => Some('r'),
        5 => Some('c'),
        6 => Some('p'),
        7 => Some('P'),
        8 => Some('C'),
        9 => Some('R'),
        10 => Some('N'),
        11 => Some('B'),
        12 => Some('A'),
        13 => Some('K'),
        _ => None,
    }
}

fn legal_dense_mask(position: &Position) -> Vec<bool> {
    let mut out = vec![false; DENSE_MOVE_SPACE];
    for move_index in legal_dense_moves(position) {
        out[move_index] = true;
    }
    out
}

fn legal_dense_moves(position: &Position) -> Vec<usize> {
    position
        .legal_moves()
        .into_iter()
        .map(dense_move_index)
        .collect()
}

fn unpack_legal_mask_row(mask: &ndarray::Array<u8, Ix2>, row: usize) -> Vec<bool> {
    let mut out = vec![false; DENSE_MOVE_SPACE];
    for index in 0..DENSE_MOVE_SPACE {
        let byte = mask[[row, index / 8]];
        let bit = 7 - (index % 8);
        out[index] = ((byte >> bit) & 1) != 0;
    }
    out
}

fn read_f16_npz_2d(path: &Path, name: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut zip = ZipArchive::new(file)?;
    let fallback = format!("{name}.npy");
    let member_index = zip
        .index_for_name(name)
        .or_else(|| zip.index_for_name(&fallback))
        .ok_or_else(|| format!("missing npz member `{name}`"))?;
    let mut member = zip.by_index(member_index)?;
    let mut bytes = Vec::new();
    member.read_to_end(&mut bytes)?;
    read_f16_npy_2d(&bytes)
}

fn read_f16_npy_2d(bytes: &[u8]) -> Result<Array2<f32>, Box<dyn Error>> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("invalid .npy header".into());
    }
    let major = bytes[6];
    let header_start;
    let header_len;
    match major {
        1 => {
            header_start = 10usize;
            header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        }
        2 | 3 => {
            header_start = 12usize;
            header_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        }
        _ => return Err(format!("unsupported .npy version {major}").into()),
    }
    let data_start = header_start + header_len;
    if data_start > bytes.len() {
        return Err("truncated .npy header".into());
    }
    let header = std::str::from_utf8(&bytes[header_start..data_start])?;
    if !header.contains("'descr': '<f2'") && !header.contains("\"descr\": \"<f2\"") {
        return Err(
            format!("expected little-endian float16 policy_prob, header={header:?}").into(),
        );
    }
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        return Err("fortran-order float16 arrays are not supported".into());
    }
    let shape = parse_npy_2d_shape(header)?;
    let value_count = shape.0 * shape.1;
    let byte_count = value_count * 2;
    if data_start + byte_count > bytes.len() {
        return Err("truncated float16 .npy payload".into());
    }
    let mut values = Vec::with_capacity(value_count);
    for chunk in bytes[data_start..data_start + byte_count].chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        values.push(half::f16::from_bits(bits).to_f32());
    }
    Ok(Array2::from_shape_vec((shape.0, shape.1), values)?)
}

fn parse_npy_2d_shape(header: &str) -> Result<(usize, usize), Box<dyn Error>> {
    let shape_pos = header.find("shape").ok_or("missing shape in .npy header")?;
    let after_shape = &header[shape_pos..];
    let start = after_shape
        .find('(')
        .ok_or("missing shape tuple in .npy header")?;
    let end = after_shape[start + 1..]
        .find(')')
        .ok_or("unterminated shape tuple in .npy header")?
        + start
        + 1;
    let dims = after_shape[start + 1..end]
        .split(',')
        .filter_map(|part| {
            let part = part.trim();
            (!part.is_empty()).then(|| part.parse::<usize>())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dims.len() != 2 {
        return Err(format!("expected 2D shape, got {dims:?}").into());
    }
    Ok((dims[0], dims[1]))
}
