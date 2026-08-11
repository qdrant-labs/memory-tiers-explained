use qdrant_client::{
    Qdrant, QdrantError,
    qdrant::{
        BoolIndexParamsBuilder, CollectionExistsRequest, CreateCollectionBuilder,
        CreateFieldIndexCollectionBuilder, DatetimeIndexParamsBuilder, Distance, FieldType,
        FloatIndexParamsBuilder, GeoIndexParamsBuilder, HnswConfigDiffBuilder,
        IntegerIndexParamsBuilder, KeywordIndexParamsBuilder, Memory, OptimizersConfigDiffBuilder,
        PayloadStorageParamsBuilder, QuantizationType, ScalarQuantizationBuilder,
        SparseIndexConfigBuilder, SparseVectorConfig, SparseVectorParamsBuilder,
        SparseVectorsConfigBuilder, TextIndexParamsBuilder, TokenizerType, TurboQuantBitSize,
        TurboQuantizationBuilder, UuidIndexParamsBuilder, VectorParamsBuilder, VectorsConfig,
        VectorsConfigBuilder,
    },
};
use std::{
    collections::HashMap,
    env::{VarError, var as env_var},
    str::FromStr,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SetupError {
    #[error("Invalid memory tier: {0}")]
    InvalidMemoryTierError(String),
    #[error("Invalid field type for payload indexing: {0}")]
    InvalidFieldTypeError(String),
    #[error(transparent)]
    EnvVarNotFoundError(#[from] VarError),
    #[error(transparent)]
    QdrantError(#[from] QdrantError),
    #[error("Collection already exists")]
    CollectionAlreadyExistsError,
    #[error("Invalid field key format: expected 'key=type', got: {0}")]
    InvalidFieldKeyFormat(String),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MemoryTier {
    Cached,
    Cold,
    Pinned,
}

impl MemoryTier {
    fn into_memory(self) -> Memory {
        match self {
            Self::Cached => Memory::Cached,
            Self::Cold => Memory::Cold,
            Self::Pinned => Memory::Pinned,
        }
    }
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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PayloadFieldType {
    Keyword,
    Integer,
    Float,
    Geo,
    Text,
    Bool,
    Datetime,
    Uuid,
}

impl FromStr for PayloadFieldType {
    type Err = SetupError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "keyword" => Ok(Self::Keyword),
            "integer" | "int" => Ok(Self::Integer),
            "boolean" | "bool" => Ok(Self::Bool),
            "datetime" => Ok(Self::Datetime),
            "float" => Ok(Self::Float),
            "geo" | "geospatial" => Ok(Self::Geo),
            "text" => Ok(Self::Text),
            "uuid" => Ok(Self::Uuid),
            _ => Err(SetupError::InvalidFieldTypeError(s.to_owned())),
        }
    }
}

impl PayloadFieldType {
    fn into_field_type(&self) -> FieldType {
        match self {
            Self::Bool => FieldType::Bool,
            Self::Datetime => FieldType::Datetime,
            Self::Float => FieldType::Float,
            Self::Geo => FieldType::Geo,
            Self::Integer => FieldType::Integer,
            Self::Keyword => FieldType::Keyword,
            Self::Text => FieldType::Text,
            Self::Uuid => FieldType::Uuid,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HnswTierConfig {
    pub memory_tier: Option<MemoryTier>,
    pub use_inline_storage: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QuantizationTierConfig {
    pub memory_tier: Option<MemoryTier>,
    pub quantized: bool,
    pub use_turboquant: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryTiersSetup {
    pub dense_vectors: Option<MemoryTier>,
    pub hnsw_config: Option<HnswTierConfig>,
    pub quantization: Option<QuantizationTierConfig>,
    pub sparse_vector_index: Option<MemoryTier>,
    pub use_sparse: bool,
    pub payloads: Option<MemoryTier>,
    pub payload_index: Option<MemoryTier>,
    pub mmap_threshold: Option<u64>,
    pub indexing_theshold: Option<u64>,
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

    pub async fn create_collection(
        &self,
        collection_name: &str,
        payload_keys: Option<HashMap<String, PayloadFieldType>>,
        memory_tiers_setup: MemoryTiersSetup,
        verbose: bool,
    ) -> Result<(), SetupError> {
        if verbose {
            println!(
                "Creating collection {} with the following settings:\n{:#?}",
                collection_name, memory_tiers_setup
            );
        }
        let client = Qdrant::from_url(&self.qdrant_api_url)
            .api_key(self.qdrant_api_key.as_deref())
            .build()?;
        if verbose {
            println!("Checking collection existence");
        }
        let coll_exists = client
            .collection_exists(CollectionExistsRequest {
                collection_name: collection_name.to_owned(),
            })
            .await?;
        if coll_exists {
            return Err(SetupError::CollectionAlreadyExistsError);
        }
        let mut collection_builder = CreateCollectionBuilder::new(collection_name);
        let mut optimizer_config: Option<OptimizersConfigDiffBuilder> = None;
        if let Some(mt) = memory_tiers_setup.mmap_threshold {
            optimizer_config = Some(OptimizersConfigDiffBuilder::default().memmap_threshold(mt));
        }
        if let Some(it) = memory_tiers_setup.indexing_theshold {
            optimizer_config = Some(optimizer_config.map_or(
                OptimizersConfigDiffBuilder::default().indexing_threshold(it),
                |p| p.indexing_threshold(it),
            ));
        }
        if let Some(oc) = optimizer_config {
            collection_builder = collection_builder.optimizers_config(oc.build());
        }
        if let Some(dv) = memory_tiers_setup.dense_vectors {
            let mut vectors_builder =
                VectorParamsBuilder::new(1024, Distance::Cosine).memory(dv.into_memory());
            if let Some(qt) = memory_tiers_setup.quantization {
                if qt.quantized && qt.use_turboquant {
                    let mut quant_builder =
                        TurboQuantizationBuilder::default().bits(TurboQuantBitSize::Bits4);
                    if let Some(mem) = qt.memory_tier {
                        quant_builder = quant_builder.memory(mem.into_memory());
                    }
                    vectors_builder = vectors_builder.quantization_config(quant_builder.build())
                } else if qt.quantized && !qt.use_turboquant {
                    let mut quant_builder =
                        ScalarQuantizationBuilder::default().r#type(QuantizationType::Int8.into());
                    if let Some(mem) = qt.memory_tier {
                        quant_builder = quant_builder.memory(mem.into_memory());
                    }
                    vectors_builder = vectors_builder.quantization_config(quant_builder.build())
                }
            }
            collection_builder = collection_builder.vectors_config(VectorsConfig::from(
                VectorsConfigBuilder::default()
                    .add_named_vector_params("dense", vectors_builder.build())
                    .to_owned(),
            ));
        } else {
            let mut vectors_builder = VectorParamsBuilder::new(1024, Distance::Cosine);
            if let Some(qt) = memory_tiers_setup.quantization {
                if qt.quantized && qt.use_turboquant {
                    let mut quant_builder =
                        TurboQuantizationBuilder::default().bits(TurboQuantBitSize::Bits4);
                    if let Some(mem) = qt.memory_tier {
                        quant_builder = quant_builder.memory(mem.into_memory());
                    }
                    vectors_builder = vectors_builder.quantization_config(quant_builder.build())
                } else if qt.quantized && !qt.use_turboquant {
                    let mut quant_builder =
                        ScalarQuantizationBuilder::default().r#type(QuantizationType::Int8.into());
                    if let Some(mem) = qt.memory_tier {
                        quant_builder = quant_builder.memory(mem.into_memory());
                    }
                    vectors_builder = vectors_builder.quantization_config(quant_builder.build())
                }
            }
            collection_builder = collection_builder.vectors_config(vectors_builder.build());
        }
        if let Some(hwc) = memory_tiers_setup.hnsw_config {
            if hwc.use_inline_storage {
                collection_builder = collection_builder
                    .hnsw_config(HnswConfigDiffBuilder::default().inline_storage(true))
            } else if let Some(mt) = hwc.memory_tier {
                collection_builder = collection_builder
                    .hnsw_config(HnswConfigDiffBuilder::default().memory(mt.into_memory()))
            }
        }
        if memory_tiers_setup.use_sparse {
            if let Some(sp) = memory_tiers_setup.sparse_vector_index {
                collection_builder =
                    collection_builder.sparse_vectors_config(SparseVectorConfig::from(
                        SparseVectorsConfigBuilder::default()
                            .add_named_vector_params(
                                "sparse",
                                SparseVectorParamsBuilder::default().index(
                                    SparseIndexConfigBuilder::default().memory(sp.into_memory()),
                                ),
                            )
                            .to_owned(),
                    ))
            } else {
                collection_builder =
                    collection_builder.sparse_vectors_config(SparseVectorConfig::from(
                        SparseVectorsConfigBuilder::default()
                            .add_named_vector_params("sparse", SparseVectorParamsBuilder::default())
                            .to_owned(),
                    ))
            }
        }
        if let Some(pay) = memory_tiers_setup.payloads {
            collection_builder = collection_builder.payload(
                PayloadStorageParamsBuilder::default()
                    .memory(pay.into_memory())
                    .build(),
            )
        }
        client.create_collection(collection_builder.build()).await?;
        if verbose {
            println!("Created collection");
        }
        if let Some(keys) = payload_keys
            && !keys.is_empty()
        {
            if verbose {
                println!(
                    "Creating field indexes for the following fields: {}",
                    keys.keys().cloned().collect::<Vec<String>>().join(", ")
                );
            }
            for (k, tp) in keys {
                let mut req_builder = CreateFieldIndexCollectionBuilder::new(
                    collection_name,
                    k,
                    tp.into_field_type(),
                )
                .wait(true);
                if let Some(pi) = memory_tiers_setup.payload_index {
                    match tp {
                        PayloadFieldType::Bool => {
                            req_builder = req_builder.field_index_params(
                                BoolIndexParamsBuilder::new().memory(pi.into_memory()),
                            )
                        }
                        PayloadFieldType::Datetime => {
                            req_builder = req_builder.field_index_params(
                                DatetimeIndexParamsBuilder::default().memory(pi.into_memory()),
                            )
                        }
                        PayloadFieldType::Float => {
                            req_builder = req_builder.field_index_params(
                                FloatIndexParamsBuilder::new().memory(pi.into_memory()),
                            )
                        }
                        PayloadFieldType::Geo => {
                            req_builder = req_builder.field_index_params(
                                GeoIndexParamsBuilder::new().memory(pi.into_memory()),
                            )
                        }
                        PayloadFieldType::Integer => {
                            req_builder = req_builder.field_index_params(
                                IntegerIndexParamsBuilder::default().memory(pi.into_memory()),
                            )
                        }
                        PayloadFieldType::Keyword => {
                            req_builder = req_builder.field_index_params(
                                KeywordIndexParamsBuilder::default().memory(pi.into_memory()),
                            )
                        }
                        PayloadFieldType::Text => {
                            req_builder = req_builder.field_index_params(
                                TextIndexParamsBuilder::new(TokenizerType::Word)
                                    .memory(pi.into_memory()),
                            )
                        }
                        PayloadFieldType::Uuid => {
                            req_builder = req_builder.field_index_params(
                                UuidIndexParamsBuilder::default().memory(pi.into_memory()),
                            )
                        }
                    }
                }
                client.create_field_index(req_builder.build()).await?;
            }
        }
        if verbose {
            println!("Created field indexes");
        }
        Ok(())
    }
}
