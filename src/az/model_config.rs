use std::io;

use super::{
    CNN_CHANNELS, POLICY_CONDITION_SIZE, VALUE_BRANCH_DEPTH, VALUE_BRANCH_SIZE, VALUE_HIDDEN_SIZE,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AzModelConfig {
    pub hidden_size: usize,
    pub cnn_channels: usize,
    pub value_branch_size: usize,
    pub value_branch_depth: usize,
    pub value_hidden_size: usize,
    pub policy_condition_size: usize,
    pub attention_feedback: bool,
}

impl Default for AzModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 256,
            cnn_channels: CNN_CHANNELS,
            value_branch_size: VALUE_BRANCH_SIZE,
            value_branch_depth: VALUE_BRANCH_DEPTH,
            value_hidden_size: VALUE_HIDDEN_SIZE,
            policy_condition_size: POLICY_CONDITION_SIZE,
            attention_feedback: true,
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
        self.cnn_channels = self.cnn_channels.max(1);
        self.value_branch_size = self.value_branch_size.max(1);
        self.value_branch_depth = self.value_branch_depth.max(1);
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
                "unsupported model config {:?}; this build currently supports hidden_size only, with cnn_channels={}, value_branch_size={}, value_branch_depth={}, value_hidden_size={}, policy_condition_size={}, attention_feedback=true",
                config,
                CNN_CHANNELS,
                VALUE_BRANCH_SIZE,
                VALUE_BRANCH_DEPTH,
                VALUE_HIDDEN_SIZE,
                POLICY_CONDITION_SIZE
            ),
        ))
    }
}
