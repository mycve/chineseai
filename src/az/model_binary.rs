use std::fs;
use std::io::{self, BufWriter, Cursor, Read, Write};
use std::path::Path;

use super::model::VALUE_RELATION_LAYERS;
use super::model::{AZ_MODEL_BINARY_HEADER_LEN, AZ_MODEL_BINARY_MAGIC, AZ_MODEL_BINARY_VERSION};
use super::{
    AzModel, AzModelConfig, BOARD_CHANNELS, BOARD_INPUT_KERNEL_AREA, BOARD_PLANES_SIZE,
    DENSE_MOVE_SPACE, VALUE_LOGITS, mobile_block_bias_size, mobile_block_weight_size,
    value_head_features, value_relation_bias_size, value_relation_weight_size,
};

fn write_f32_slice_le<W: Write>(writer: &mut W, slice: &[f32]) -> io::Result<()> {
    for &value in slice {
        writer.write_all(&value.to_bits().to_le_bytes())?;
    }
    Ok(())
}

fn read_u32_le(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32_vec_le(reader: &mut impl Read, len: usize) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0f32; len];
    let mut buf = [0u8; 4];
    for slot in &mut out {
        reader.read_exact(&mut buf)?;
        *slot = f32::from_bits(u32::from_le_bytes(buf));
    }
    Ok(out)
}

impl AzModel {
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let file = fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(AZ_MODEL_BINARY_MAGIC)?;
        writer.write_all(&AZ_MODEL_BINARY_VERSION.to_le_bytes())?;
        writer.write_all(&(BOARD_CHANNELS as u32).to_le_bytes())?;
        writer.write_all(&(self.hidden_size as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.model_channels as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.model_blocks as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.value_head_channels as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.value_hidden_size as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.policy_condition_size as u32).to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?;
        write_f32_slice_le(&mut writer, &self.board_conv1_weights)?;
        write_f32_slice_le(&mut writer, &self.board_conv1_bias)?;
        write_f32_slice_le(&mut writer, &self.board_conv2_weights)?;
        write_f32_slice_le(&mut writer, &self.board_conv2_bias)?;
        write_f32_slice_le(&mut writer, &self.board_bn_scale)?;
        write_f32_slice_le(&mut writer, &self.board_bn_bias)?;
        write_f32_slice_le(&mut writer, &self.board_bn_running_mean)?;
        write_f32_slice_le(&mut writer, &self.board_bn_running_var)?;
        write_f32_slice_le(&mut writer, &self.residual_bn_scale)?;
        write_f32_slice_le(&mut writer, &self.residual_bn_bias)?;
        write_f32_slice_le(&mut writer, &self.residual_bn_running_mean)?;
        write_f32_slice_le(&mut writer, &self.residual_bn_running_var)?;
        write_f32_slice_le(&mut writer, &self.position_embed)?;
        write_f32_slice_le(&mut writer, &self.value_relation_weights)?;
        write_f32_slice_le(&mut writer, &self.value_relation_bias)?;
        write_f32_slice_le(&mut writer, &self.value_tail_conv_weights)?;
        write_f32_slice_le(&mut writer, &self.value_tail_conv_bias)?;
        write_f32_slice_le(&mut writer, &self.value_tail_bn_scale)?;
        write_f32_slice_le(&mut writer, &self.value_tail_bn_bias)?;
        write_f32_slice_le(&mut writer, &self.value_tail_bn_running_mean)?;
        write_f32_slice_le(&mut writer, &self.value_tail_bn_running_var)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_hidden)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_bias)?;
        write_f32_slice_le(&mut writer, &self.value_logits_weights)?;
        write_f32_slice_le(&mut writer, &self.value_direct_logits_weights)?;
        write_f32_slice_le(&mut writer, &self.value_logits_bias)?;
        write_f32_slice_le(&mut writer, &self.value_scalar_hidden_weights)?;
        write_f32_slice_le(&mut writer, &self.value_scalar_direct_weights)?;
        write_f32_slice_le(&mut writer, &self.value_scalar_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_tail_conv_weights)?;
        write_f32_slice_le(&mut writer, &self.policy_tail_conv_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_tail_bn_scale)?;
        write_f32_slice_le(&mut writer, &self.policy_tail_bn_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_tail_bn_running_mean)?;
        write_f32_slice_le(&mut writer, &self.policy_tail_bn_running_var)?;
        write_f32_slice_le(&mut writer, &self.policy_logits_weights)?;
        write_f32_slice_le(&mut writer, &self.policy_move_bias)?;
        writer.flush()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path.as_ref())?;
        Self::decode_binary(&bytes)
    }

    fn decode_binary(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < AZ_MODEL_BINARY_HEADER_LEN || !bytes.starts_with(AZ_MODEL_BINARY_MAGIC) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated or invalid AzModel binary",
            ));
        }
        let mut reader = Cursor::new(&bytes[AZ_MODEL_BINARY_MAGIC.len()..]);
        let version = read_u32_le(&mut reader)?;
        if version != AZ_MODEL_BINARY_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported AzModel binary version {version} (expected {AZ_MODEL_BINARY_VERSION})"
                ),
            ));
        }
        let input_channels = read_u32_le(&mut reader)? as usize;
        let hidden_size = read_u32_le(&mut reader)? as usize;
        let model_config = AzModelConfig {
            hidden_size,
            model_channels: read_u32_le(&mut reader)? as usize,
            model_blocks: read_u32_le(&mut reader)? as usize,
            value_head_channels: read_u32_le(&mut reader)? as usize,
            value_hidden_size: read_u32_le(&mut reader)? as usize,
            policy_condition_size: read_u32_le(&mut reader)? as usize,
        }
        .normalized();
        let _reserved = read_u32_le(&mut reader)?;
        model_config.validate_supported()?;
        if input_channels != BOARD_CHANNELS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "binary input channel count does not match this build",
            ));
        }
        let channels = model_config.model_channels;
        let blocks = model_config.model_blocks;
        let value_channels = model_config.value_head_channels;
        let value_hidden_size = model_config.value_hidden_size;
        let value_features = value_head_features(channels, value_channels);
        let board_conv1_weights_len = channels * BOARD_CHANNELS * BOARD_INPUT_KERNEL_AREA;
        let board_conv1_bias_len = channels;
        let board_conv2_weights_len = blocks * mobile_block_weight_size(channels);
        let board_conv2_bias_len = blocks * mobile_block_bias_size(channels);
        let board_bn_len = channels;
        let residual_bn_len = blocks * mobile_block_bias_size(channels);
        let position_embed_len = channels * BOARD_PLANES_SIZE;
        let value_relation_weights_len =
            VALUE_RELATION_LAYERS * value_relation_weight_size(channels);
        let value_relation_bias_len = VALUE_RELATION_LAYERS * value_relation_bias_size(channels);
        let value_tail_conv_weights_len = value_channels * channels;
        let value_tail_conv_bias_len = value_channels;
        let value_tail_bn_len = value_channels;
        let vih_len = value_hidden_size * value_features;
        let vib_len = value_hidden_size;
        let vlw_len = VALUE_LOGITS * value_hidden_size;
        let vdlw_len = VALUE_LOGITS * value_features;
        let vlb_len = VALUE_LOGITS;
        let vshw_len = value_hidden_size;
        let vsdw_len = value_features;
        let vsb_len = 1;
        let ptcw_len = super::model::POLICY_HEAD_CHANNELS * channels;
        let ptcb_len = super::model::POLICY_HEAD_CHANNELS;
        let ptbn_len = super::model::POLICY_HEAD_CHANNELS;
        let plw_len = DENSE_MOVE_SPACE * super::model::policy_head_map_size();
        let pmb_len = DENSE_MOVE_SPACE;
        let float_count = board_conv1_weights_len
            + board_conv1_bias_len
            + board_conv2_weights_len
            + board_conv2_bias_len
            + board_bn_len * 4
            + residual_bn_len * 4
            + position_embed_len
            + value_relation_weights_len
            + value_relation_bias_len
            + value_tail_conv_weights_len
            + value_tail_conv_bias_len
            + value_tail_bn_len * 4
            + vih_len
            + vib_len
            + vlw_len
            + vdlw_len
            + vlb_len
            + vshw_len
            + vsdw_len
            + vsb_len
            + ptcw_len
            + ptcb_len
            + ptbn_len * 4
            + plw_len
            + pmb_len;
        let expected_len = AZ_MODEL_BINARY_HEADER_LEN + float_count * 4;
        if bytes.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "AzModel binary size mismatch: got {} bytes, expected {} (floats {})",
                    bytes.len(),
                    expected_len,
                    float_count
                ),
            ));
        }
        let model = Self {
            model_config,
            hidden_size,
            board_conv1_weights: read_f32_vec_le(&mut reader, board_conv1_weights_len)?,
            board_conv1_bias: read_f32_vec_le(&mut reader, board_conv1_bias_len)?,
            board_conv2_weights: read_f32_vec_le(&mut reader, board_conv2_weights_len)?,
            board_conv2_bias: read_f32_vec_le(&mut reader, board_conv2_bias_len)?,
            board_bn_scale: read_f32_vec_le(&mut reader, board_bn_len)?,
            board_bn_bias: read_f32_vec_le(&mut reader, board_bn_len)?,
            board_bn_running_mean: read_f32_vec_le(&mut reader, board_bn_len)?,
            board_bn_running_var: read_f32_vec_le(&mut reader, board_bn_len)?,
            residual_bn_scale: read_f32_vec_le(&mut reader, residual_bn_len)?,
            residual_bn_bias: read_f32_vec_le(&mut reader, residual_bn_len)?,
            residual_bn_running_mean: read_f32_vec_le(&mut reader, residual_bn_len)?,
            residual_bn_running_var: read_f32_vec_le(&mut reader, residual_bn_len)?,
            position_embed: read_f32_vec_le(&mut reader, position_embed_len)?,
            value_relation_weights: read_f32_vec_le(&mut reader, value_relation_weights_len)?,
            value_relation_bias: read_f32_vec_le(&mut reader, value_relation_bias_len)?,
            value_tail_conv_weights: read_f32_vec_le(&mut reader, value_tail_conv_weights_len)?,
            value_tail_conv_bias: read_f32_vec_le(&mut reader, value_tail_conv_bias_len)?,
            value_tail_bn_scale: read_f32_vec_le(&mut reader, value_tail_bn_len)?,
            value_tail_bn_bias: read_f32_vec_le(&mut reader, value_tail_bn_len)?,
            value_tail_bn_running_mean: read_f32_vec_le(&mut reader, value_tail_bn_len)?,
            value_tail_bn_running_var: read_f32_vec_le(&mut reader, value_tail_bn_len)?,
            value_intermediate_hidden: read_f32_vec_le(&mut reader, vih_len)?,
            value_intermediate_bias: read_f32_vec_le(&mut reader, vib_len)?,
            value_logits_weights: read_f32_vec_le(&mut reader, vlw_len)?,
            value_direct_logits_weights: read_f32_vec_le(&mut reader, vdlw_len)?,
            value_logits_bias: read_f32_vec_le(&mut reader, vlb_len)?,
            value_scalar_hidden_weights: read_f32_vec_le(&mut reader, vshw_len)?,
            value_scalar_direct_weights: read_f32_vec_le(&mut reader, vsdw_len)?,
            value_scalar_bias: read_f32_vec_le(&mut reader, vsb_len)?,
            policy_tail_conv_weights: read_f32_vec_le(&mut reader, ptcw_len)?,
            policy_tail_conv_bias: read_f32_vec_le(&mut reader, ptcb_len)?,
            policy_tail_bn_scale: read_f32_vec_le(&mut reader, ptbn_len)?,
            policy_tail_bn_bias: read_f32_vec_le(&mut reader, ptbn_len)?,
            policy_tail_bn_running_mean: read_f32_vec_le(&mut reader, ptbn_len)?,
            policy_tail_bn_running_var: read_f32_vec_le(&mut reader, ptbn_len)?,
            policy_logits_weights: read_f32_vec_le(&mut reader, plw_len)?,
            policy_move_bias: read_f32_vec_le(&mut reader, pmb_len)?,
            gpu_trainer: None,
        };
        model.validate()?;
        Ok(model)
    }
}
