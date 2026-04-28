use std::io;

use super::{
    CNN_CHANNELS, POLICY_CONDITION_SIZE, RESIDUAL_BLOCKS, VALUE_HEAD_CHANNELS, VALUE_HIDDEN_SIZE,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AzModelConfig {
    pub hidden_size: usize,
    pub model_channels: usize,
    pub model_blocks: usize,
    pub value_head_channels: usize,
    pub value_hidden_size: usize,
    pub policy_condition_size: usize,
}

impl Default for AzModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 256,
            model_channels: CNN_CHANNELS,
            model_blocks: RESIDUAL_BLOCKS,
            value_head_channels: VALUE_HEAD_CHANNELS,
            value_hidden_size: VALUE_HIDDEN_SIZE,
            policy_condition_size: POLICY_CONDITION_SIZE,
        }
    }
}

impl AzModelConfig {
    pub fn with_hidden_size(hidden_size: usize) -> Self {
        Self {
            hidden_size: hidden_size.max(1),
            ..Self::default()
        }
    }

    pub fn normalized(mut self) -> Self {
        self.hidden_size = self.hidden_size.max(1);
        self.model_channels = self.model_channels.max(1);
        self.model_blocks = self.model_blocks.max(1);
        self.value_head_channels = self.value_head_channels.max(1);
        self.value_hidden_size = self.value_hidden_size.max(1);
        self.policy_condition_size = self.policy_condition_size.max(1);
        self
    }

    pub fn validate_supported(&self) -> io::Result<()> {
        let config = self.normalized();
        if config.policy_condition_size == POLICY_CONDITION_SIZE {
            return Ok(());
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unsupported policy_condition_size {}; this field is a fixed legacy header slot and must remain {}",
                config.policy_condition_size, POLICY_CONDITION_SIZE
            ),
        ))
    }
}
