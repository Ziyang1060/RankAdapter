"""
Processes a JSONL file containing queries and hits, generates TREC-compatible output files, and indexes the corpus using Pyserini.

See usage by running:
    python scripts/jsonl_to_trec.py --help
"""
import os
import json
import subprocess
import argparse


def process_single_jsonl_file(file_path, dataset_name, max_hits, output_dir):
    # Initialize data storage
    data = {
        "corpus": [],
        "queries": [],
        "qrels": [],
        "trec_run": []
    }

    # Read and parse the JSONL file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            query = json_obj.get("query", "")
            hits = json_obj.get("hits", [])

            # Generate a unique query_id for each query
            query_id = len(data["queries"])
            data["queries"].append(f"{query_id}\t{query}")

            # Truncate hits to the specified max_hits
            if max_hits > 0:
                hits = hits[:max_hits]
            for rank, hit in enumerate(hits):
                content = hit.get("content", "")
                doc_id = f"{query_id}-{rank}"
                label = hit.get("label", 0)  # Default label is 0 if not provided

                # Add document to corpus
                data["corpus"].append((doc_id, content))

                # Add relevance judgment (qrel) line
                qrel_line = f"{query_id} Q0 {doc_id} {label}"
                data["qrels"].append(qrel_line)

                # Add TREC run line
                score = len(hits) - rank
                trec_run_line = f"{query_id} Q0 {doc_id} {rank+1} {score:.6f} Default"
                data["trec_run"].append(trec_run_line)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Write corpus.jsonl
    corpus_file_path = os.path.join(output_dir, "corpus.jsonl")
    with open(corpus_file_path, 'w', encoding='utf-8') as corpus_file:
        for doc_id, content in data["corpus"]:
            corpus_file.write(json.dumps({"id": doc_id, "contents": content}, ensure_ascii=False) + "\n")

    # Write queries.tsv
    with open(os.path.join(output_dir, "queries.tsv"), 'w', encoding='utf-8') as queries_file:
        queries_file.write("\n".join(data["queries"]) + "\n")

    # Write qrels.txt
    with open(os.path.join(output_dir, "qrels.txt"), 'w', encoding='utf-8') as qrels_file:
        qrels_file.write("\n".join(data["qrels"]) + "\n")

    # Write TREC run file
    trec_run_file_path = os.path.join('data', 'first_stage_run', f"run.custom.default.{dataset_name}.txt")
    with open(trec_run_file_path, 'w', encoding='utf-8') as trec_run_file:
        trec_run_file.write("\n".join(data["trec_run"]) + "\n")

    print("Processing completed. Output files have been generated.")
    return corpus_file_path


def execute_lucene_indexing(corpus_directory, index_dir):
    try:
        command = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", corpus_directory,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        print("Executing Lucene indexing command:")
        print(" ".join(command))
        subprocess.run(command, check=True)
        print("Lucene indexing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Lucene indexing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL file and execute Lucene indexing.")
    parser.add_argument("--file_path", type=str, help="Path to the JSONL file.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name of the dataset. Default to JSONL file name.")
    parser.add_argument("--max_hits", type=int, default=-1,
                        help="Maximum number of hits to retrieve. Default to -1, which means all hits.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to store the output files. Default to `data/custom_dataset/{dataset_name}`.")
    parser.add_argument("--index_dir", type=str, default=None,
                        help="Directory to store the Lucene index. Default to --output_dir + /index")

    args = parser.parse_args()

    if args.dataset_name is None:
        print(f"Dataset name not provided. Using JSONL file name as dataset name.")
        args.dataset_name = os.path.splitext(os.path.basename(args.file_path))[0]
    if args.output_dir is None:
        args.output_dir = os.path.join("data", "custom_dataset", args.dataset_name)
    if args.index_dir is None:
        args.index_dir = os.path.join(args.output_dir, "index")

    # Step 1: Process the JSONL file and generate the corpus
    corpus_file_path = process_single_jsonl_file(
        args.file_path,
        dataset_name=args.dataset_name,
        max_hits=args.max_hits,
        output_dir=args.output_dir
    )

    # Step 2: Execute Lucene indexing on the generated corpus
    execute_lucene_indexing(args.output_dir, index_dir=args.index_dir)