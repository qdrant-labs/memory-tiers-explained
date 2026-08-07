use datafusion::{
    arrow::{
        array::{Array, AsArray, Float32Array, Int64Array, StringArray},
        datatypes::Float32Type,
    },
    error::DataFusionError,
    execution::{config::SessionConfig, context::SessionContext, options::ParquetReadOptions},
};
use futures::StreamExt;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use thiserror::Error;

use qdrant_client::{
    Qdrant, QdrantError,
    qdrant::{
        NamedVectors, PointId, PointStruct, UpsertPointsBuilder, Value, Vector,
        point_id::PointIdOptions,
    },
};
use serde::{Deserialize, Serialize};

use crate::metrics::LatencyStats;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPayload {
    doc_id: String,
    url: String,
    title: String,
    start_char: i64,
    end_char: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmbeddingWithPayload {
    embedding: Vec<f32>,
    payload: DataPayload,
}

impl Into<HashMap<String, Value>> for DataPayload {
    fn into(self) -> HashMap<String, Value> {
        let mut map = HashMap::new();
        map.insert("doc_id".to_string(), Value::from(self.doc_id));
        map.insert("url".to_string(), Value::from(self.url));
        map.insert("title".to_string(), Value::from(self.title));
        map.insert("start_char".to_string(), Value::from(self.start_char));
        map.insert("end_char".to_string(), Value::from(self.end_char));
        map
    }
}

fn convert_to_points(to_upload: Vec<EmbeddingWithPayload>) -> Vec<PointStruct> {
    let points: Vec<PointStruct> = to_upload
        .par_iter()
        .cloned()
        .map(|t| {
            let point_id = uuid::Uuid::new_v4();
            let vector_d = Vector::new_dense(t.embedding);
            let vectors = NamedVectors::default().add_vector("dense", vector_d);
            PointStruct {
                id: Some(PointId {
                    point_id_options: Some(PointIdOptions::Uuid(point_id.to_string())),
                }),
                payload: t.payload.into(),
                vectors: Some(vectors.into()),
            }
        })
        .collect();

    points
}

#[derive(Debug, Error)]
pub enum LoadError {
    #[error(transparent)]
    QdrantError(#[from] QdrantError),
    #[error(transparent)]
    DataFusionError(#[from] DataFusionError),
}

pub async fn load_embeddings(
    path: &str,
    qdrant_api_url: &str,
    qdrant_api_key: Option<&str>,
    collection_name: &str,
    verbose: bool,
) -> Result<(), LoadError> {
    let client = Qdrant::from_url(qdrant_api_url)
        .api_key(qdrant_api_key)
        .timeout(600)
        .build()?;
    let mut session_config = SessionConfig::new();
    session_config
        .options_mut()
        .execution
        .parquet
        .schema_force_view_types = false;
    let ctx = SessionContext::new_with_config(session_config);
    ctx.register_parquet("data", path, ParquetReadOptions::default())
        .await?;

    let df = ctx
        .sql("SELECT emb, docid, url, title, start_char, end_char FROM data")
        .await?;
    let mut stream = df.execute_stream().await?;
    let mut total_points = 0;
    let mut batch_durations = Vec::new();
    let upload_start = Instant::now();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let embedding_col = batch.column(0).as_list::<i32>();
        let docid_col: &StringArray = batch.column(1).as_string();
        let url_col: &StringArray = batch.column(2).as_string();
        let title_col: &StringArray = batch.column(3).as_string();
        let start_char_col: &Int64Array = batch.column(4).as_primitive();
        let end_char_col: &Int64Array = batch.column(5).as_primitive();

        let mut results = Vec::with_capacity(embedding_col.len());

        for i in 0..embedding_col.len() {
            if embedding_col.is_null(i) {
                continue;
            }

            let values = embedding_col.value(i);
            let float_array: &Float32Array = values.as_primitive::<Float32Type>();
            let embedding: Vec<f32> = float_array.values().to_vec();

            let payload = DataPayload {
                doc_id: docid_col.value(i).to_string(),
                url: url_col.value(i).to_string(),
                title: title_col.value(i).to_string(),
                start_char: start_char_col.value(i),
                end_char: end_char_col.value(i),
            };

            results.push(EmbeddingWithPayload { embedding, payload });
        }

        // Upload this batch directly, awaited in-loop
        let points = convert_to_points(results);
        total_points += points.len() as u32;
        if verbose {
            println!("Uploading {:?} points to {}", points.len(), collection_name);
        }
        let batch_start = Instant::now();
        client
            .upsert_points(UpsertPointsBuilder::new(collection_name, points).build())
            .await?;
        batch_durations.push(batch_start.elapsed());
    }
    let upload_elapsed = upload_start.elapsed();

    let latency = LatencyStats::from_durations(batch_durations, upload_elapsed);
    println!(
        "Upload done! Total points: {total_points}, total batches: {}, total upload time: {:.3}s ({:.1} batches/s)",
        latency.count,
        latency.wall_clock.as_secs_f64(),
        latency.throughput(),
    );
    println!(
        "batch upload latency: min={:?} p50/median={:?} p95={:?} p99={:?} max={:?} mean={:?}",
        latency.min, latency.p50, latency.p95, latency.p99, latency.max, latency.mean,
    );

    Ok(())
}
