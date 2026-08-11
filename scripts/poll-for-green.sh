#! /bin/bash

COLLECTION_NAME="$1"
url="${QDRANT_URL//\:6334/\:6333}"
QUERIES_FILE="$HOME/.cache/huggingface/hub/datasets--CohereLabs--msmarco-v2.1-embed-english-v3/snapshots/e78737fe92ac1b783211b705c12207ca75fcc9b7/queries_parquet/queries.parquet"

echo "Starting to poll for green status"

while true
do
    if [ -z $QDRANT_API_KEY ]; then
        status=$(curl -s ${url}/collections/${COLLECTION_NAME} | jq -r ".result.status")
        if [[ "$status" == "green" ]]; then
            break
        else
            sleep 5
        fi
    else
        status=$(curl -s ${url}/collections/${COLLECTION_NAME} -H "api-key: ${QDRANT_API_KEY}" | jq -r ".result.status")
        if [[ "$status" == "green" ]]; then
            break
        else
            sleep 5
        fi
    fi
done

cargo run --bin benchmark -- search $QUERIES_FILE $COLLECTION_NAME --verbose
