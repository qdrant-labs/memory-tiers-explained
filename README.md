# Qdrant Memory Tiers

Repository to understand the performance and trade-offs of Qdrant's per-component memory tier
settings (`cached`, `cold`, `pinned`) across dense vectors, HNSW graphs, quantized vectors,
payloads, payload indexes, and sparse vector indexes.

The workflow is: spin up Qdrant, create a collection with a chosen memory-tier layout, upload the
MS MARCO passage embeddings, run search benchmarks, and record the results for comparison.

## Repo layout

- `crates/collection-setup` — CLI to create a Qdrant collection with a specific memory-tier
  configuration (which storage tier each component lives in, quantization, payload indexes, etc).
- `crates/benchmark` — CLI to upload embeddings from Parquet files into a collection and to run
  search benchmarks (sequential or concurrent) against it, reporting latency/throughput stats.
- `packages/data-download` — Python package (uv) that downloads the benchmark dataset from
  Hugging Face.
- `configs/` — Qdrant server config overlays used by the two `compose.yaml` variants (`base.yaml`,
  `io-uring.yaml`).
- `scripts/` — shell scripts that orchestrate full end-to-end benchmark runs (setup → upload → wait
  for green → search → teardown) for different memory-tier presets.
- `results/` — captured output (`config.txt`, `upload.txt`, `metrics.txt`) from past benchmark
  runs, organized by dataset size and preset name.
- `compose.yaml` / `compose.io-uring.yaml` — local Qdrant via Docker Compose, with/without
  `io_uring` async scoring enabled.

## Prerequisites

- Rust toolchain (edition 2024) + Cargo
- Docker (for a local Qdrant instance) — or a Qdrant Cloud instance
- [uv](https://docs.astral.sh/uv/) for the `data-download` Python package
- `jq` and `curl` (used by `scripts/poll-for-green.sh` and `scripts/delete-collection.sh`)

## Configuration

Create a `.env` file (or otherwise export) with the target Qdrant instance's connection details:

```
QDRANT_URL="https://<host>:6334"
QDRANT_API_KEY="<api-key>"
```

All CLIs and scripts fall back to these env vars when `--url`/`--api-key` (or
`--base-url`/`--api-key`) aren't passed explicitly. `QDRANT_API_KEY` can be omitted for a local,
unauthenticated Qdrant.

## Getting started

### 1. Start Qdrant

```bash
docker compose -f compose.yaml up -d              # base config
# or
docker compose -f compose.io-uring.yaml up -d      # with io_uring async scorer enabled
```

### 2. Download the dataset

The benchmarks use [`CohereLabs/msmarco-v2.1-embed-english-v3`](https://huggingface.co/datasets/CohereLabs/msmarco-v2.1-embed-english-v3).

```bash
cd packages/data-download
uv run data-download          # downloads both passages and queries
uv run data-download data     # passages only
uv run data-download queries  # queries only
```

This populates the local Hugging Face cache (`~/.cache/huggingface/hub/...`), which is what the
`scripts/*.sh` files expect.

### 3. Create a collection

```bash
cargo run --bin collection-setup -- <collection_name> [OPTIONS]
```

Key options (see `crates/collection-setup/src/main.rs` for the full list):

| Flag | Purpose |
| --- | --- |
| `--dense-vectors-memory <tier>` | Memory tier for the dense vector storage |
| `--hnsw-memory <tier>` | Memory tier for the HNSW graph |
| `--hnsw-inline-storage` | Store the HNSW graph inline instead of a separate tier |
| `--quantize` | Enable quantization (scalar int8) |
| `--quantized-vectors-memory <tier>` | Memory tier for the quantized vectors |
| `--use-sparse` / `--sparse-vector-index-memory <tier>` | Enable a sparse vector index and its tier |
| `--payload-memory <tier>` | Memory tier for payload storage |
| `--payload-index-memory <tier>` | Memory tier for payload field indexes |
| `--field-to-index <key=type>` | Create a payload index for `key` (repeatable); types: `keyword`, `integer`, `float`, `geo`, `text`, `bool`, `datetime`, `uuid` |
| `--mmap-threshold` / `--indexing-threshold` | Optimizer thresholds |
| `--base-url` / `--api-key` / `--verbose` | Connection + logging |

`<tier>` is one of `cached`, `cold`, or `pinned`.

### 4. Upload embeddings

```bash
cargo run --bin benchmark -- upload <parquet_path> <collection_name> [--url ...] [--api-key ...] [--verbose]
```

### 5. Run search benchmarks

```bash
cargo run --bin benchmark -- search <query_parquet_path> <collection_name> [--vector-name dense] [--limit 10] [--verbose]
cargo run --bin benchmark -- search-concurrent <query_parquet_path> <collection_name> [--concurrency 8] ...
```

Both print latency stats (min/mean/p50/p95/p99/max) and throughput.

### 6. Full benchmark presets

`scripts/` wires steps 3–5 together end-to-end for a given preset, then polls until the collection
is `green`, runs the search benchmark, and deletes the collection:

- `full-cached-no-quantization.sh <num_parquet_files>` — everything in the `cached` tier
- `full-cached-w-quantization.sh <num_parquet_files>` — `cached` tier + scalar/TurboQuant quantization
- `full-cold-no-quantization.sh <num_parquet_files>` — everything in the `cold` tier, HNSW inline storage
- `full-cold-w-quantization.sh <num_parquet_files>` — `cold` tier + quantization
- `high-speed-recommendation.sh <num_parquet_files>` — `cold` dense vectors/HNSW with `pinned` quantized vectors

Each takes the number of `msmarco_v2.1_doc_segmented_NN.parquet` files to upload (0-indexed).
Standalone helpers: `scripts/poll-for-green.sh <collection>` and
`scripts/delete-collection.sh <collection>`.

## Results

`results/<dataset-size>/<preset-name>/` captures, per run:

- `config.txt` — the `MemoryTiersSetup` used to create the collection
- `upload.txt` — upload throughput/latency summary
- `metrics.txt` — search benchmark latency/throughput summary
