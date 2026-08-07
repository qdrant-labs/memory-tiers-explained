mod metrics;
mod search;
mod upload;

use clap::{Parser, Subcommand};

use crate::search::{query_performance, query_performance_concurrent};
use crate::upload::load_embeddings;

#[derive(Debug, Subcommand)]
enum Commands {
    Upload {
        parquet_path: String,
        collection_name: String,
        #[arg(long, default_value = None)]
        api_key: Option<String>,
        #[arg(long, default_value = None)]
        url: Option<String>,
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
    Search {
        query_parquet_path: String,
        collection_name: String,
        #[arg(long, default_value = None)]
        api_key: Option<String>,
        #[arg(long, default_value = None)]
        url: Option<String>,
        #[arg(long, default_value = "dense")]
        vector_name: String,
        #[arg(long, default_value_t = 10)]
        limit: u64,
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
    SearchConcurrent {
        query_parquet_path: String,
        collection_name: String,
        #[arg(long, default_value = None)]
        api_key: Option<String>,
        #[arg(long, default_value = None)]
        url: Option<String>,
        #[arg(long, default_value = "dense")]
        vector_name: String,
        #[arg(long, default_value_t = 10)]
        limit: u64,
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
}

#[derive(Parser)]
struct CliArgs {
    #[command(subcommand)]
    cmd: Commands,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();
    match args.cmd {
        Commands::Upload {
            parquet_path,
            collection_name,
            api_key,
            url,
            verbose,
        } => {
            let qdrant_url = match url {
                Some(s) => s,
                None => std::env::var("QDRANT_URL")?,
            };
            let qdrant_api_key = match api_key {
                Some(s) => Some(s),
                None => std::env::var("QDRANT_API_KEY").ok(),
            };
            load_embeddings(
                &parquet_path,
                &qdrant_url,
                qdrant_api_key.as_deref(),
                &collection_name,
                verbose,
            )
            .await?;
        }
        Commands::Search {
            query_parquet_path,
            collection_name,
            api_key,
            url,
            vector_name,
            limit,
            verbose,
        } => {
            let qdrant_url = match url {
                Some(s) => s,
                None => std::env::var("QDRANT_URL")?,
            };
            let qdrant_api_key = match api_key {
                Some(s) => Some(s),
                None => std::env::var("QDRANT_API_KEY").ok(),
            };
            let report = query_performance(
                &qdrant_url,
                qdrant_api_key.as_deref(),
                &collection_name,
                &query_parquet_path,
                &vector_name,
                limit,
                verbose,
            )
            .await?;
            report.print();
        }
        Commands::SearchConcurrent {
            query_parquet_path,
            collection_name,
            api_key,
            url,
            vector_name,
            limit,
            concurrency,
            verbose,
        } => {
            let qdrant_url = match url {
                Some(s) => s,
                None => std::env::var("QDRANT_URL")?,
            };
            let qdrant_api_key = match api_key {
                Some(s) => Some(s),
                None => std::env::var("QDRANT_API_KEY").ok(),
            };
            let report = query_performance_concurrent(
                &qdrant_url,
                qdrant_api_key.as_deref(),
                &collection_name,
                &query_parquet_path,
                &vector_name,
                limit,
                concurrency,
                verbose,
            )
            .await?;
            report.print();
        }
    }
    Ok(())
}
