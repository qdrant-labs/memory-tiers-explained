#! /bin/bash

set -euo pipefail

COLLECTION_NAME="memory_tiers_bench_collection"
NUM_PARQUET_FILES="$1"
PARQUET_PREFIX="$HOME/.cache/huggingface/hub/datasets--CohereLabs--msmarco-v2.1-embed-english-v3/snapshots/e78737fe92ac1b783211b705c12207ca75fcc9b7/passages_parquet"

cargo run --bin collection-setup -- \
    $COLLECTION_NAME \
    --verbose \
    --dense-vectors-memory cold \
    --hnsw-memory cold \
    --hnsw-inline-storage \
    --quantize \
    --quantized-vectors-memory pinned \
    --field-to-index doc_id=keyword \
    --field-to-index url=keyword \
    --field-to-index title=keyword \
    --field-to-index start_char=int \
    --field-to-index end_char=int

for num in $(seq 0 "$NUM_PARQUET_FILES")
do
    if [ "$num" -ge 10 ]; then
        file="${PARQUET_PREFIX}/msmarco_v2.1_doc_segmented_${num}.parquet"
        cargo run --bin benchmark -- upload $file $COLLECTION_NAME --verbose
    else
        file="${PARQUET_PREFIX}/msmarco_v2.1_doc_segmented_0${num}.parquet"
        cargo run --bin benchmark -- upload $file $COLLECTION_NAME --verbose
    fi
done

./scripts/poll-for-green.sh $COLLECTION_NAME
./scripts/delete-collection.sh $COLLECTION_NAME
