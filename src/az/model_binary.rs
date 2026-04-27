use std::fs;
use std::io::{self, BufWriter, Cursor, Read, Write};
use std::path::Path;

use super::{
    AZ_MODEL_BINARY_HEADER_LEN, AZ_MODEL_BINARY_MAGIC, AZ_MODEL_BINARY_VERSION, AzModel,
    AzModelConfig, BOARD_CHANNELS, BOARD_INPUT_KERNEL_AREA, CNN_CHANNELS, CNN_KERNEL_AREA,
    CNN_POOLED_SIZE, DENSE_MOVE_SPACE, POLICY_CONDITION_SIZE, VALUE_BRANCH_DEPTH,
    VALUE_BRANCH_SIZE, VALUE_CNN_CHANNELS, VALUE_CNN_POOLED_SIZE, VALUE_HIDDEN_SIZE, VALUE_LOGITS,
    VALUE_SQUARE_INPUT_SIZE,
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
        writer.write_all(&(VALUE_SQUARE_INPUT_SIZE as u32).to_le_bytes())?;
        writer.write_all(&(self.hidden_size as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.cnn_channels as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.value_branch_size as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.value_branch_depth as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.value_hidden_size as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.policy_condition_size as u32).to_le_bytes())?;
        writer.write_all(&(self.model_config.attention_feedback as u32).to_le_bytes())?;
        write_f32_slice_le(&mut writer, &self.board_conv1_weights)?;
        write_f32_slice_le(&mut writer, &self.board_conv1_bias)?;
        write_f32_slice_le(&mut writer, &self.board_conv2_weights)?;
        write_f32_slice_le(&mut writer, &self.board_conv2_bias)?;
        write_f32_slice_le(&mut writer, &self.board_attention_query)?;
        write_f32_slice_le(&mut writer, &self.board_context_weights)?;
        write_f32_slice_le(&mut writer, &self.board_context_bias)?;
        write_f32_slice_le(&mut writer, &self.board_hidden)?;
        write_f32_slice_le(&mut writer, &self.board_hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.value_trunk_weights)?;
        write_f32_slice_le(&mut writer, &self.value_trunk_biases)?;
        write_f32_slice_le(&mut writer, &self.value_square_hidden)?;
        write_f32_slice_le(&mut writer, &self.value_square_hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.value_tail_conv_weights)?;
        write_f32_slice_le(&mut writer, &self.value_tail_conv_bias)?;
        write_f32_slice_le(&mut writer, &self.value_board_attention_query)?;
        write_f32_slice_le(&mut writer, &self.value_context_weights)?;
        write_f32_slice_le(&mut writer, &self.value_context_bias)?;
        write_f32_slice_le(&mut writer, &self.value_board_hidden)?;
        write_f32_slice_le(&mut writer, &self.value_board_hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_hidden)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_bias)?;
        write_f32_slice_le(&mut writer, &self.value_logits_weights)?;
        write_f32_slice_le(&mut writer, &self.value_logits_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_move_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_move_cnn)?;
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
        let input_size = read_u32_le(&mut reader)? as usize;
        let hidden_size = read_u32_le(&mut reader)? as usize;
        let model_config = AzModelConfig {
            hidden_size,
            cnn_channels: read_u32_le(&mut reader)? as usize,
            value_branch_size: read_u32_le(&mut reader)? as usize,
            value_branch_depth: read_u32_le(&mut reader)? as usize,
            value_hidden_size: read_u32_le(&mut reader)? as usize,
            policy_condition_size: read_u32_le(&mut reader)? as usize,
            attention_feedback: read_u32_le(&mut reader)? != 0,
        }
        .normalized();
        model_config.validate_supported()?;
        if input_size != VALUE_SQUARE_INPUT_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "binary input_size does not match this build (VALUE_SQUARE_INPUT_SIZE)",
            ));
        }
        let board_conv1_weights_len = CNN_CHANNELS * BOARD_CHANNELS * BOARD_INPUT_KERNEL_AREA;
        let board_conv1_bias_len = CNN_CHANNELS;
        let board_conv2_weights_len = CNN_CHANNELS * CNN_CHANNELS * CNN_KERNEL_AREA;
        let board_conv2_bias_len = CNN_CHANNELS;
        let board_attention_query_len = CNN_CHANNELS;
        let board_context_weights_len = CNN_CHANNELS * CNN_CHANNELS;
        let board_context_bias_len = CNN_CHANNELS;
        let board_hidden_len = hidden_size * CNN_POOLED_SIZE;
        let board_hidden_bias_len = hidden_size;
        let value_trunk_weights_len = VALUE_BRANCH_DEPTH * VALUE_BRANCH_SIZE * VALUE_BRANCH_SIZE;
        let value_trunk_biases_len = VALUE_BRANCH_DEPTH * VALUE_BRANCH_SIZE;
        let value_square_hidden_len = VALUE_SQUARE_INPUT_SIZE * VALUE_BRANCH_SIZE;
        let value_square_hidden_bias_len = VALUE_BRANCH_SIZE;
        let value_tail_conv_weights_len = VALUE_CNN_CHANNELS * CNN_CHANNELS * CNN_KERNEL_AREA;
        let value_tail_conv_bias_len = VALUE_CNN_CHANNELS;
        let value_board_attention_query_len = VALUE_CNN_CHANNELS;
        let value_context_weights_len = VALUE_CNN_CHANNELS * VALUE_CNN_CHANNELS;
        let value_context_bias_len = VALUE_CNN_CHANNELS;
        let value_board_hidden_len = VALUE_BRANCH_SIZE * VALUE_CNN_POOLED_SIZE;
        let value_board_hidden_bias_len = VALUE_BRANCH_SIZE;
        let vih_len = VALUE_HIDDEN_SIZE * VALUE_BRANCH_SIZE;
        let vib_len = VALUE_HIDDEN_SIZE;
        let vlw_len = VALUE_LOGITS * VALUE_HIDDEN_SIZE;
        let vlb_len = VALUE_LOGITS;
        let pmh_len = DENSE_MOVE_SPACE * hidden_size;
        let pmc_len = DENSE_MOVE_SPACE * CNN_POOLED_SIZE;
        let pmb_len = DENSE_MOVE_SPACE;
        let pfh_len = POLICY_CONDITION_SIZE * hidden_size;
        let pfc_len = POLICY_CONDITION_SIZE * CNN_POOLED_SIZE;
        let pfb_len = POLICY_CONDITION_SIZE;
        let float_count = board_conv1_weights_len
            + board_conv1_bias_len
            + board_conv2_weights_len
            + board_conv2_bias_len
            + board_attention_query_len
            + board_context_weights_len
            + board_context_bias_len
            + board_hidden_len
            + board_hidden_bias_len
            + value_trunk_weights_len
            + value_trunk_biases_len
            + value_square_hidden_len
            + value_square_hidden_bias_len
            + value_tail_conv_weights_len
            + value_tail_conv_bias_len
            + value_board_attention_query_len
            + value_context_weights_len
            + value_context_bias_len
            + value_board_hidden_len
            + value_board_hidden_bias_len
            + vih_len
            + vib_len
            + vlw_len
            + vlb_len
            + pmh_len
            + pmc_len
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
        let board_conv1_weights = read_f32_vec_le(&mut reader, board_conv1_weights_len)?;
        let board_conv1_bias = read_f32_vec_le(&mut reader, board_conv1_bias_len)?;
        let board_conv2_weights = read_f32_vec_le(&mut reader, board_conv2_weights_len)?;
        let board_conv2_bias = read_f32_vec_le(&mut reader, board_conv2_bias_len)?;
        let board_attention_query = read_f32_vec_le(&mut reader, board_attention_query_len)?;
        let board_context_weights = read_f32_vec_le(&mut reader, board_context_weights_len)?;
        let board_context_bias = read_f32_vec_le(&mut reader, board_context_bias_len)?;
        let board_hidden = read_f32_vec_le(&mut reader, board_hidden_len)?;
        let board_hidden_bias = read_f32_vec_le(&mut reader, board_hidden_bias_len)?;
        let value_trunk_weights = read_f32_vec_le(&mut reader, value_trunk_weights_len)?;
        let value_trunk_biases = read_f32_vec_le(&mut reader, value_trunk_biases_len)?;
        let value_square_hidden = read_f32_vec_le(&mut reader, value_square_hidden_len)?;
        let value_square_hidden_bias = read_f32_vec_le(&mut reader, value_square_hidden_bias_len)?;
        let value_tail_conv_weights = read_f32_vec_le(&mut reader, value_tail_conv_weights_len)?;
        let value_tail_conv_bias = read_f32_vec_le(&mut reader, value_tail_conv_bias_len)?;
        let value_board_attention_query =
            read_f32_vec_le(&mut reader, value_board_attention_query_len)?;
        let value_context_weights = read_f32_vec_le(&mut reader, value_context_weights_len)?;
        let value_context_bias = read_f32_vec_le(&mut reader, value_context_bias_len)?;
        let value_board_hidden = read_f32_vec_le(&mut reader, value_board_hidden_len)?;
        let value_board_hidden_bias = read_f32_vec_le(&mut reader, value_board_hidden_bias_len)?;
        let value_intermediate_hidden = read_f32_vec_le(&mut reader, vih_len)?;
        let value_intermediate_bias = read_f32_vec_le(&mut reader, vib_len)?;
        let value_logits_weights = read_f32_vec_le(&mut reader, vlw_len)?;
        let value_logits_bias = read_f32_vec_le(&mut reader, vlb_len)?;
        let policy_move_hidden = read_f32_vec_le(&mut reader, pmh_len)?;
        let policy_move_cnn = read_f32_vec_le(&mut reader, pmc_len)?;
        let policy_move_bias = read_f32_vec_le(&mut reader, pmb_len)?;
        let policy_feature_hidden = read_f32_vec_le(&mut reader, pfh_len)?;
        let policy_feature_cnn = read_f32_vec_le(&mut reader, pfc_len)?;
        let policy_feature_bias = read_f32_vec_le(&mut reader, pfb_len)?;
        let model = Self {
            model_config,
            hidden_size,
            board_conv1_weights,
            board_conv1_bias,
            board_conv2_weights,
            board_conv2_bias,
            board_attention_query,
            board_context_weights,
            board_context_bias,
            board_hidden,
            board_hidden_bias,
            value_trunk_weights,
            value_trunk_biases,
            value_square_hidden,
            value_square_hidden_bias,
            value_tail_conv_weights,
            value_tail_conv_bias,
            value_board_attention_query,
            value_context_weights,
            value_context_bias,
            value_board_hidden,
            value_board_hidden_bias,
            value_intermediate_hidden,
            value_intermediate_bias,
            value_logits_weights,
            value_logits_bias,
            policy_move_hidden,
            policy_move_cnn,
            policy_move_bias,
            policy_feature_hidden,
            policy_feature_cnn,
            policy_feature_bias,
            gpu_trainer: None,
        };
        model.validate()?;
        Ok(model)
    }
}
