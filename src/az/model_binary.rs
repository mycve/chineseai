use std::fs;
use std::io::{self, BufWriter, Cursor, Read, Write};
use std::path::Path;

use super::{
    AZ_MODEL_BINARY_HEADER_LEN, AZ_MODEL_BINARY_MAGIC, AZ_MODEL_BINARY_VERSION, AzModel,
    AzModelConfig, BOARD_CHANNELS, BOARD_INPUT_KERNEL_AREA, BOARD_PLANES_SIZE, CNN_CHANNELS,
    CNN_KERNEL_AREA, CNN_POOLED_SIZE, DENSE_MOVE_SPACE, POLICY_CONDITION_SIZE, RESIDUAL_BLOCKS,
    VALUE_HEAD_CHANNELS, VALUE_HEAD_FEATURES, VALUE_HIDDEN_SIZE, VALUE_LOGITS,
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
        writer.write_all(&(self.model_config.cnn_channels as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.residual_blocks as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.value_head_channels as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.value_hidden_size as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.policy_condition_size as u32).to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?;
        write_f32_slice_le(&mut writer, &self.board_conv1_weights)?;
        write_f32_slice_le(&mut writer, &self.board_conv1_bias)?;
        write_f32_slice_le(&mut writer, &self.board_conv2_weights)?;
        write_f32_slice_le(&mut writer, &self.board_conv2_bias)?;
        write_f32_slice_le(&mut writer, &self.position_embed)?;
        write_f32_slice_le(&mut writer, &self.line_gates)?;
        write_f32_slice_le(&mut writer, &self.board_hidden)?;
        write_f32_slice_le(&mut writer, &self.board_hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.value_tail_conv_weights)?;
        write_f32_slice_le(&mut writer, &self.value_tail_conv_bias)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_hidden)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_bias)?;
        write_f32_slice_le(&mut writer, &self.value_logits_weights)?;
        write_f32_slice_le(&mut writer, &self.value_direct_logits_weights)?;
        write_f32_slice_le(&mut writer, &self.value_logits_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_from_weights)?;
        write_f32_slice_le(&mut writer, &self.policy_from_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_to_weights)?;
        write_f32_slice_le(&mut writer, &self.policy_to_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_move_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_feature_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_feature_cnn)?;
        write_f32_slice_le(&mut writer, &self.policy_feature_bias)?;
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
            cnn_channels: read_u32_le(&mut reader)? as usize,
            residual_blocks: read_u32_le(&mut reader)? as usize,
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
        let board_conv1_weights_len = CNN_CHANNELS * BOARD_CHANNELS * BOARD_INPUT_KERNEL_AREA;
        let board_conv1_bias_len = CNN_CHANNELS;
        let board_conv2_weights_len =
            RESIDUAL_BLOCKS * 2 * CNN_CHANNELS * CNN_CHANNELS * CNN_KERNEL_AREA;
        let board_conv2_bias_len = RESIDUAL_BLOCKS * 2 * CNN_CHANNELS;
        let position_embed_len = CNN_CHANNELS * BOARD_PLANES_SIZE;
        let line_gates_len = RESIDUAL_BLOCKS * 2 * CNN_CHANNELS;
        let board_hidden_len = hidden_size * CNN_POOLED_SIZE;
        let board_hidden_bias_len = hidden_size;
        let value_tail_conv_weights_len = VALUE_HEAD_CHANNELS * CNN_CHANNELS;
        let value_tail_conv_bias_len = VALUE_HEAD_CHANNELS;
        let vih_len = VALUE_HIDDEN_SIZE * VALUE_HEAD_FEATURES;
        let vib_len = VALUE_HIDDEN_SIZE;
        let vlw_len = VALUE_LOGITS * VALUE_HIDDEN_SIZE;
        let vdlw_len = VALUE_LOGITS * VALUE_HEAD_FEATURES;
        let vlb_len = VALUE_LOGITS;
        let pfw_len = CNN_CHANNELS;
        let pfbias_len = 1;
        let ptw_len = CNN_CHANNELS;
        let ptbias_len = 1;
        let pmb_len = DENSE_MOVE_SPACE;
        let pfh_len = POLICY_CONDITION_SIZE * hidden_size;
        let pfc_len = POLICY_CONDITION_SIZE * CNN_POOLED_SIZE;
        let pfb_len = POLICY_CONDITION_SIZE;
        let float_count = board_conv1_weights_len
            + board_conv1_bias_len
            + board_conv2_weights_len
            + board_conv2_bias_len
            + position_embed_len
            + line_gates_len
            + board_hidden_len
            + board_hidden_bias_len
            + value_tail_conv_weights_len
            + value_tail_conv_bias_len
            + vih_len
            + vib_len
            + vlw_len
            + vdlw_len
            + vlb_len
            + pfw_len
            + pfbias_len
            + ptw_len
            + ptbias_len
            + pmb_len
            + pfh_len
            + pfc_len
            + pfb_len;
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
            position_embed: read_f32_vec_le(&mut reader, position_embed_len)?,
            line_gates: read_f32_vec_le(&mut reader, line_gates_len)?,
            board_hidden: read_f32_vec_le(&mut reader, board_hidden_len)?,
            board_hidden_bias: read_f32_vec_le(&mut reader, board_hidden_bias_len)?,
            value_tail_conv_weights: read_f32_vec_le(&mut reader, value_tail_conv_weights_len)?,
            value_tail_conv_bias: read_f32_vec_le(&mut reader, value_tail_conv_bias_len)?,
            value_intermediate_hidden: read_f32_vec_le(&mut reader, vih_len)?,
            value_intermediate_bias: read_f32_vec_le(&mut reader, vib_len)?,
            value_logits_weights: read_f32_vec_le(&mut reader, vlw_len)?,
            value_direct_logits_weights: read_f32_vec_le(&mut reader, vdlw_len)?,
            value_logits_bias: read_f32_vec_le(&mut reader, vlb_len)?,
            policy_from_weights: read_f32_vec_le(&mut reader, pfw_len)?,
            policy_from_bias: read_f32_vec_le(&mut reader, pfbias_len)?,
            policy_to_weights: read_f32_vec_le(&mut reader, ptw_len)?,
            policy_to_bias: read_f32_vec_le(&mut reader, ptbias_len)?,
            policy_move_bias: read_f32_vec_le(&mut reader, pmb_len)?,
            policy_feature_hidden: read_f32_vec_le(&mut reader, pfh_len)?,
            policy_feature_cnn: read_f32_vec_le(&mut reader, pfc_len)?,
            policy_feature_bias: read_f32_vec_le(&mut reader, pfb_len)?,
            gpu_trainer: None,
        };
        model.validate()?;
        Ok(model)
    }
}
