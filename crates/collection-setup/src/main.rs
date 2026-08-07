mod setup;

use std::{collections::HashMap, env, str::FromStr};

use clap::Parser;

use crate::setup::{
    CollectionSetupWiz, HnswTierConfig, MemoryTier, MemoryTiersSetup, PayloadFieldType,
    QuantizationTierConfig, SetupError,
};

#[derive(Debug, Parser)]
struct CliArgs {
    collection_name: String,
    #[arg(long, default_value = None)]
    dense_vectors_memory: Option<String>,
    #[arg(long, default_value_t = false)]
    hnsw_inline_storage: bool,
    #[arg(long, default_value = None)]
    hnsw_memory: Option<String>,
    #[arg(long, default_value_t = false)]
    quantize: bool,
    #[arg(long, default_value = None)]
    quantized_vectors_memory: Option<String>,
    #[arg(long, default_value = None)]
    sparse_vector_index_memory: Option<String>,
    #[arg(long, default_value_t = false)]
    use_sparse: bool,
    #[arg(long, default_value = None)]
    payload_memory: Option<String>,
    #[arg(long, default_value = None)]
    payload_index_memory: Option<String>,
    #[arg(long, default_value = None)]
    mmap_threshold: Option<u64>,
    #[arg(long, default_value = None)]
    indexing_threshold: Option<u64>,
    #[arg(long, default_value_t = false)]
    io_uring: bool,
    #[arg(long)]
    field_to_index: Vec<String>,
    #[arg(long, default_value = None)]
    base_url: Option<String>,
    #[arg(long, default_value = None)]
    api_key: Option<String>,
    #[arg(long, short, default_value_t = false)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<(), SetupError> {
    let args = CliArgs::parse();

    let wiz = if args.base_url.is_none() && args.api_key.is_none() {
        CollectionSetupWiz::try_from_env()?
    } else {
        let base_url = if let Some(bu) = args.base_url {
            bu
        } else {
            env::var("QDRANT_URL")?
        };
        CollectionSetupWiz::new(args.api_key, base_url)
    };
    let hnsw_config = HnswTierConfig {
        memory_tier: {
            match args.hnsw_memory {
                None => None,
                Some(s) => Some(MemoryTier::from_str(&s)?),
            }
        },
        use_inline_storage: args.hnsw_inline_storage,
    };
    let quantization_config = QuantizationTierConfig {
        memory_tier: {
            match args.quantized_vectors_memory {
                None => None,
                Some(s) => Some(MemoryTier::from_str(&s)?),
            }
        },
        quantized: args.quantize,
    };
    let setup = MemoryTiersSetup {
        hnsw_config: Some(hnsw_config),
        quantization: Some(quantization_config),
        payload_index: match args.payload_index_memory {
            None => None,
            Some(s) => Some(MemoryTier::from_str(&s)?),
        },
        payloads: match args.payload_memory {
            None => None,
            Some(s) => Some(MemoryTier::from_str(&s)?),
        },
        sparse_vector_index: match args.sparse_vector_index_memory {
            None => None,
            Some(s) => Some(MemoryTier::from_str(&s)?),
        },
        dense_vectors: match args.dense_vectors_memory {
            None => None,
            Some(s) => Some(MemoryTier::from_str(&s)?),
        },
        mmap_threshold: args.mmap_threshold,
        indexing_theshold: args.indexing_threshold,
        use_sparse: args.use_sparse,
    };
    let mut to_index: HashMap<String, PayloadFieldType> = HashMap::new();
    for k in args.field_to_index {
        let split = k.split_once("=");
        if let Some((k, v)) = split {
            to_index.insert(k.to_owned(), PayloadFieldType::from_str(v)?);
        } else {
            return Err(SetupError::InvalidFieldKeyFormat(k));
        }
    }
    wiz.create_collection(&args.collection_name, Some(to_index), setup, args.verbose)
        .await?;

    Ok(())
}
