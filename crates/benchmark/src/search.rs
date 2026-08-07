use std::time::{Duration, Instant};

use datafusion::{
    arrow::{
        array::{Array, AsArray, Float32Array},
        datatypes::Float32Type,
    },
    execution::{config::SessionConfig, context::SessionContext, options::ParquetReadOptions},
};
use futures::stream::{self, StreamExt};
use qdrant_client::{
    Qdrant,
    qdrant::{HardwareUsage, QueryPointsBuilder},
};
use serde::Deserialize;

use crate::metrics::LatencyStats;
use crate::upload::LoadError;

pub type SearchError = LoadError;

async fn load_query_embeddings(path: &str) -> Result<Vec<Vec<f32>>, SearchError> {
    let mut session_config = SessionConfig::new();
    session_config
        .options_mut()
        .execution
        .parquet
        .schema_force_view_types = false;
    let ctx = SessionContext::new_with_config(session_config);
    ctx.register_parquet("data", path, ParquetReadOptions::default())
        .await?;
    let df = ctx.sql("select emb from data").await?;
    let batches = df.collect().await?;
    let mut embeddings: Vec<Vec<f32>> = vec![];
    for batch in &batches {
        let column = batch.column(0);
        let list_array = column.as_list::<i32>();

        for i in 0..list_array.len() {
            if list_array.is_null(i) {
                continue;
            }
            let values = list_array.value(i);
            let float_array: &Float32Array = values.as_primitive::<Float32Type>();
            let vec: Vec<f32> = float_array.values().to_vec();
            embeddings.push(vec);
        }
    }

    Ok(embeddings)
}

/// Proxy for collection storage/memory pressure, taken from the gRPC collection-info
/// response. It doesn't expose raw RAM/disk byte counts (those live behind Qdrant's
/// REST telemetry endpoints, outside this client) - segment/point/index counts are
/// what's available here.
#[derive(Debug, Clone, Copy, Default)]
pub struct CollectionSnapshot {
    pub segments_count: u64,
    pub points_count: Option<u64>,
    pub indexed_vectors_count: Option<u64>,
}

async fn snapshot_collection(
    client: &Qdrant,
    collection_name: &str,
) -> Result<CollectionSnapshot, SearchError> {
    let info = client.collection_info(collection_name).await?;
    let result = info.result.unwrap_or_default();
    Ok(CollectionSnapshot {
        segments_count: result.segments_count,
        points_count: result.points_count,
        indexed_vectors_count: result.indexed_vectors_count,
    })
}

/// Global process memory (jemalloc stats), read from the REST `/telemetry`
/// endpoint - the gRPC API this benchmark otherwise uses doesn't expose it.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct TelemetryMemory {
    pub active_bytes: u64,
    pub allocated_bytes: u64,
    pub metadata_bytes: u64,
    pub resident_bytes: u64,
    pub retained_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct TelemetryResult {
    memory: Option<TelemetryMemory>,
}

#[derive(Debug, Deserialize)]
struct TelemetryResponse {
    result: TelemetryResult,
}

/// The gRPC client talks to port 6334; telemetry is only exposed over REST,
/// on port 6333.
fn grpc_url_to_rest_url(qdrant_api_url: &str) -> String {
    qdrant_api_url.replace(":6334", ":6333")
}

/// Best-effort: telemetry is a diagnostic extra, so a REST-side failure (e.g.
/// no REST access from this deployment) shouldn't fail the whole benchmark.
async fn fetch_telemetry_memory(
    qdrant_api_url: &str,
    qdrant_api_key: Option<&str>,
) -> Option<TelemetryMemory> {
    let rest_url = grpc_url_to_rest_url(qdrant_api_url);
    let mut request = reqwest::Client::new().get(format!("{rest_url}/telemetry?details_level=3"));
    if let Some(key) = qdrant_api_key {
        request = request.header("api-key", key);
    }
    let response = match request.send().await {
        Ok(response) => response,
        Err(err) => {
            eprintln!("Warning: failed to fetch telemetry from {rest_url}: {err}");
            return None;
        }
    };
    match response.json::<TelemetryResponse>().await {
        Ok(body) => body.result.memory,
        Err(err) => {
            eprintln!("Warning: failed to parse telemetry response from {rest_url}: {err}");
            None
        }
    }
}

/// Sum of the per-query `usage.hardware` counters Qdrant reports on the query
/// response (only populated when hardware reporting is enabled server-side).
#[derive(Debug, Clone, Copy, Default)]
pub struct HardwareUsageTotals {
    pub cpu: u64,
    pub payload_io_read: u64,
    pub payload_io_write: u64,
    pub payload_index_io_read: u64,
    pub payload_index_io_write: u64,
    pub vector_io_read: u64,
    pub vector_io_write: u64,
    /// Number of queries that actually reported hardware usage. If this is 0
    /// while `count` in the latency stats isn't, hardware reporting is off.
    pub samples: usize,
}

impl HardwareUsageTotals {
    fn add(&mut self, hw: &HardwareUsage) {
        self.cpu += hw.cpu;
        self.payload_io_read += hw.payload_io_read;
        self.payload_io_write += hw.payload_io_write;
        self.payload_index_io_read += hw.payload_index_io_read;
        self.payload_index_io_write += hw.payload_index_io_write;
        self.vector_io_read += hw.vector_io_read;
        self.vector_io_write += hw.vector_io_write;
        self.samples += 1;
    }
}

struct QuerySample {
    duration: Duration,
    hardware: Option<HardwareUsage>,
}

fn split_samples(samples: Vec<QuerySample>) -> (Vec<Duration>, HardwareUsageTotals) {
    let mut durations = Vec::with_capacity(samples.len());
    let mut hardware_usage = HardwareUsageTotals::default();
    for sample in samples {
        durations.push(sample.duration);
        if let Some(hw) = &sample.hardware {
            hardware_usage.add(hw);
        }
    }
    (durations, hardware_usage)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QueryPerformanceReport {
    pub latency: LatencyStats,
    pub hardware_usage: HardwareUsageTotals,
    pub collection_before: CollectionSnapshot,
    pub collection_after: CollectionSnapshot,
    pub memory_before: Option<TelemetryMemory>,
    pub memory_after: Option<TelemetryMemory>,
}

impl QueryPerformanceReport {
    pub fn print(&self) {
        let l = &self.latency;
        println!(
            "queries={} wall={:.3}s qps={:.1} | latency min={:?} p50={:?} p95={:?} p99={:?} max={:?} mean={:?}",
            l.count,
            l.wall_clock.as_secs_f64(),
            l.throughput(),
            l.min,
            l.p50,
            l.p95,
            l.p99,
            l.max,
            l.mean,
        );
        println!(
            "collection segments: {} -> {}",
            self.collection_before.segments_count, self.collection_after.segments_count
        );
        println!(
            "collection points: {:?} -> {:?}",
            self.collection_before.points_count, self.collection_after.points_count
        );
        println!(
            "collection indexed_vectors: {:?} -> {:?}",
            self.collection_before.indexed_vectors_count,
            self.collection_after.indexed_vectors_count
        );
        let hw = &self.hardware_usage;
        if hw.samples == 0 {
            println!("hardware usage: not reported by server (hardware reporting disabled?)");
        } else {
            println!(
                "hardware usage (summed over {} queries): cpu={} payload_io_read={} payload_io_write={} payload_index_io_read={} payload_index_io_write={} vector_io_read={} vector_io_write={}",
                hw.samples,
                hw.cpu,
                hw.payload_io_read,
                hw.payload_io_write,
                hw.payload_index_io_read,
                hw.payload_index_io_write,
                hw.vector_io_read,
                hw.vector_io_write,
            );
        }
        match (self.memory_before, self.memory_after) {
            (Some(before), Some(after)) => {
                println!(
                    "server memory (jemalloc): resident {} -> {} bytes (delta {}), active {} -> {}, allocated {} -> {}, metadata {} -> {}, retained {} -> {}",
                    before.resident_bytes,
                    after.resident_bytes,
                    after.resident_bytes as i64 - before.resident_bytes as i64,
                    before.active_bytes,
                    after.active_bytes,
                    before.allocated_bytes,
                    after.allocated_bytes,
                    before.metadata_bytes,
                    after.metadata_bytes,
                    before.retained_bytes,
                    after.retained_bytes,
                );
            }
            _ => println!(
                "server memory: telemetry unavailable (no REST access, or memory not reported)"
            ),
        }
    }
}

/// Runs each query embedding through the Qdrant client one at a time, timing every
/// request individually.
pub async fn query_performance(
    qdrant_api_url: &str,
    qdrant_api_key: Option<&str>,
    collection_name: &str,
    query_embeddings_path: &str,
    vector_name: &str,
    limit: u64,
    verbose: bool,
) -> Result<QueryPerformanceReport, SearchError> {
    let client = Qdrant::from_url(qdrant_api_url)
        .api_key(qdrant_api_key)
        .timeout(600)
        .build()?;

    let embeddings = load_query_embeddings(query_embeddings_path).await?;
    if verbose {
        println!("Loaded {} query embeddings", embeddings.len());
    }

    let collection_before = snapshot_collection(&client, collection_name).await?;
    let memory_before = fetch_telemetry_memory(qdrant_api_url, qdrant_api_key).await;

    let mut samples = Vec::with_capacity(embeddings.len());
    let wall_start = Instant::now();
    for embedding in embeddings {
        let start = Instant::now();
        let response = client
            .query(
                QueryPointsBuilder::new(collection_name)
                    .query(embedding)
                    .using(vector_name)
                    .limit(limit),
            )
            .await?;
        let elapsed = start.elapsed();
        if verbose {
            println!(
                "query returned {} points in {:?}",
                response.result.len(),
                elapsed
            );
        }
        samples.push(QuerySample {
            duration: elapsed,
            hardware: response.usage.and_then(|u| u.hardware),
        });
    }
    let wall_clock = wall_start.elapsed();

    let collection_after = snapshot_collection(&client, collection_name).await?;
    let memory_after = fetch_telemetry_memory(qdrant_api_url, qdrant_api_key).await;

    let (durations, hardware_usage) = split_samples(samples);

    Ok(QueryPerformanceReport {
        latency: LatencyStats::from_durations(durations, wall_clock),
        hardware_usage,
        collection_before,
        collection_after,
        memory_before,
        memory_after,
    })
}

/// Same benchmark as [`query_performance`], but dispatches up to `concurrency` queries
/// at once instead of waiting for each one to finish, to put more read pressure on the
/// server.
pub async fn query_performance_concurrent(
    qdrant_api_url: &str,
    qdrant_api_key: Option<&str>,
    collection_name: &str,
    query_embeddings_path: &str,
    vector_name: &str,
    limit: u64,
    concurrency: usize,
    verbose: bool,
) -> Result<QueryPerformanceReport, SearchError> {
    let client = Qdrant::from_url(qdrant_api_url)
        .api_key(qdrant_api_key)
        .build()?;

    let embeddings = load_query_embeddings(query_embeddings_path).await?;
    if verbose {
        println!(
            "Loaded {} query embeddings, dispatching with concurrency={}",
            embeddings.len(),
            concurrency
        );
    }

    let collection_before = snapshot_collection(&client, collection_name).await?;
    let memory_before = fetch_telemetry_memory(qdrant_api_url, qdrant_api_key).await;

    let client_ref = &client;
    let wall_start = Instant::now();
    let samples: Vec<QuerySample> = stream::iter(embeddings)
        .map(|embedding| async move {
            let start = Instant::now();
            let response = client_ref
                .query(
                    QueryPointsBuilder::new(collection_name)
                        .query(embedding)
                        .using(vector_name)
                        .limit(limit),
                )
                .await?;
            let elapsed = start.elapsed();
            if verbose {
                println!(
                    "query returned {} points in {:?}",
                    response.result.len(),
                    elapsed
                );
            }
            Ok::<QuerySample, SearchError>(QuerySample {
                duration: elapsed,
                hardware: response.usage.and_then(|u| u.hardware),
            })
        })
        .buffer_unordered(concurrency)
        .collect::<Vec<Result<QuerySample, SearchError>>>()
        .await
        .into_iter()
        .collect::<Result<Vec<QuerySample>, SearchError>>()?;
    let wall_clock = wall_start.elapsed();

    let collection_after = snapshot_collection(&client, collection_name).await?;
    let memory_after = fetch_telemetry_memory(qdrant_api_url, qdrant_api_key).await;

    let (durations, hardware_usage) = split_samples(samples);

    Ok(QueryPerformanceReport {
        latency: LatencyStats::from_durations(durations, wall_clock),
        hardware_usage,
        collection_before,
        collection_after,
        memory_before,
        memory_after,
    })
}
