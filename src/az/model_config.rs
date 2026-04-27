use std::io;

use super::{
    CNN_CHANNELS, POLICY_CONDITION_SIZE, RESIDUAL_BLOCKS, VALUE_HEAD_CHANNELS, VALUE_HIDDEN_SIZE,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AzModelConfig {
    pub hidden_size: usize,
    pub line_channels: usize,
    pub line_blocks: usize,
    pub value_head_channels: usize,
    pub value_hidden_size: usize,
    pub policy_condition_size: usize,
}

impl Default for AzModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 256,
            line_channels: CNN_CHANNELS,
            line_blocks: RESIDUAL_BLOCKS,
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
        self.line_channels = self.line_channels.max(1);
        self.line_blocks = self.line_blocks.max(1);
        self.value_head_channels = self.value_head_channels.max(1);
        self.value_hidden_size = self.value_hidden_size.max(1);
        self.policy_condition_size = self.policy_condition_size.max(1);
        self
    }

    pub fn validate_supported(&self) -> io::Result<()> {
        let config = self.normalized();
        let expected = Self::with_hidden_size(config.hidden_size);
        if config == expected {
            return Ok(());
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unsupported model config {:?}; this build currently supports hidden_size only, with line_channels={}, line_blocks={}, value_head_channels={}, value_hidden_size={}, policy_condition_size={}",
                config,
                CNN_CHANNELS,
                RESIDUAL_BLOCKS,
                VALUE_HEAD_CHANNELS,
                VALUE_HIDDEN_SIZE,
                POLICY_CONDITION_SIZE
            ),
        ))
    }
}
