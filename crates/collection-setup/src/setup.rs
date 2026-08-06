use std::{
    env::{VarError, var as env_var},
    str::FromStr,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SetupError {
    #[error("Invalid memory tier: {0}")]
    InvalidMemoryTierError(String),
    #[error(transparent)]
    EnvVarNotFoundError(#[from] VarError),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MemoryTier {
    Cached,
    Cold,
    Pinned,
}

impl FromStr for MemoryTier {
    type Err = SetupError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cold" => Ok(Self::Cold),
            "cached" => Ok(Self::Cached),
            "pinned" => Ok(Self::Pinned),
            _ => Err(SetupError::InvalidMemoryTierError(s.to_owned())),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HnswTierConfig {
    memory_tier: Option<MemoryTier>,
    use_inline_storage: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryTiersSetup {
    pub dense_vectors: Option<MemoryTier>,
    pub hnsw_config: Option<HnswTierConfig>,
    pub quantized_vectors: Option<MemoryTier>,
    pub sparse_vector_index: Option<MemoryTier>,
    pub payloads: Option<MemoryTier>,
    pub payload_index: Option<MemoryTier>,
}

#[derive(Debug, Clone)]
pub struct CollectionSetupWiz {
    pub qdrant_api_key: Option<String>,
    pub qdrant_api_url: String,
}

impl CollectionSetupWiz {
    pub fn new(qdrant_api_key: Option<String>, qdrant_api_url: String) -> Self {
        Self {
            qdrant_api_key,
            qdrant_api_url,
        }
    }

    pub fn try_from_env() -> Result<Self, SetupError> {
        let qdrant_api_url = env_var("QDRANT_URL")?;
        let qdrant_api_key = env_var("QDRANT_API_KEY").ok();
        Ok(Self {
            qdrant_api_key,
            qdrant_api_url,
        })
    }

    pub fn create_collection(&self) -> Result<(), SetupError> {
        Ok(())
    }
}
