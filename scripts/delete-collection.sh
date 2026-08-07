#! /bin/bash

COLLECTION_NAME="$1"
url="${QDRANT_URL//\:6334/\:6333}"


if [ -z $QDRANT_API_KEY ]; then
    curl -X DELETE -s ${url}/collections/${COLLECTION_NAME}
else
    curl -X DELETE -s ${url}/collections/${COLLECTION_NAME} -H "api_key: ${QDRANT_API_KEY}"
fi

echo ""
echo "Deleted collection"
