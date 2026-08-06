import sys

from huggingface_hub import snapshot_download


def download_dataset() -> None:
    snapshot_download(
        "CohereLabs/msmarco-v2.1-embed-english-v3",
        repo_type="dataset",
        allow_patterns=[
            "passages_parquet/msmarco_v2.1_doc_segmented_00.parquet",
            "passages_parquet/msmarco_v2.1_doc_segmented_01.parquet",
            "passages_parquet/msmarco_v2.1_doc_segmented_02.parquet",
            "passages_parquet/msmarco_v2.1_doc_segmented_03.parquet",
            "passages_parquet/msmarco_v2.1_doc_segmented_04.parquet",
            "passages_parquet/msmarco_v2.1_doc_segmented_05.parquet",
            "passages_parquet/msmarco_v2.1_doc_segmented_06.parquet",
            "passages_parquet/msmarco_v2.1_doc_segmented_07.parquet",
            "passages_parquet/msmarco_v2.1_doc_segmented_08.parquet",
            "passages_parquet/msmarco_v2.1_doc_segmented_09.parquet",
            "passages_parquet/msmarco_v2.1_doc_segmented_10.parquet",
        ],
    )


def download_queries() -> None:
    snapshot_download(
        "CohereLabs/msmarco-v2.1-embed-english-v3",
        repo_type="dataset",
        allow_patterns=[
            "queries_parquet/queries.parquet",
        ],
    )

def main() -> None:
    args = sys.argv
    if len(args) == 1:
        download_dataset()
        download_queries()
    if args[1] == "queries":
        download_queries()
    elif args[1] == "data":
        download_dataset()
    else:
        print(f"Unsupported download target: {args[1]}")
        sys.exit(1)
