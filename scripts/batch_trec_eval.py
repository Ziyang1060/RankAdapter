import csv
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define evaluation command templates
commands = {
    "ndcg_cut_5": "python -m pyserini.eval.trec_eval -c -m ndcg_cut.5 {qrels} {run_file}",
    "ndcg_cut_10": "python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {qrels} {run_file}",
    "recall_10": "python -m pyserini.eval.trec_eval -c -m recall.10 {qrels} {run_file}",
    "map_10": "python -m pyserini.eval.trec_eval -c -M 10 -m map {qrels} {run_file}"
}

# Regular expression to extract evaluation results
pattern = re.compile(r"all\s+([0-9.]+)")

# Path to save the evaluation results
output_csv = "evaluation_results.csv"

# Templates for qrels file paths
custom_qrels_template = "data/custom_dataset/{dataset_name}/qrels.txt"
beir_qrels_template = {
    "trec-covid": "beir-v1.0.0-trec-covid-test",
    "webis-touche": "beir-v1.0.0-webis-touche2020-test",
    # Add more BEIR datasets as needed
}

# Directory containing run files
run_dir = "data/run/"
run_files = [f for f in os.listdir(run_dir) if f.startswith("run.")]

# Container to store evaluation results
results = []

# Infer dataset type and qrels based on the filename
def infer_dataset_and_qrels(run_file):
    dataset_type = run_file.split(".")[1]
    dataset_name = run_file.split(".")[-2]
    if dataset_type == "custom":
        qrels = custom_qrels_template.format(dataset_name=dataset_name)
    elif dataset_type == "beir":
        qrels = beir_qrels_template[dataset_name]
    else:
        raise ValueError(f"Unknown dataset type for file: {run_file}")
    return dataset_type, dataset_name, qrels

# Function to evaluate a single run file
def evaluate_file(run_file):
    run_path = os.path.join(run_dir, run_file)
    print(f"Processing {run_path}...")

    # Infer dataset information
    try:
        dataset_type, dataset_name, qrels = infer_dataset_and_qrels(run_file)
    except ValueError as e:
        print(e)
        return None

    # Initialize the result dictionary
    result = {"file": run_file, "dataset_type": dataset_type, "dataset_name": dataset_name}

    # Iterate over evaluation commands
    for metric, command_template in commands.items():
        command = command_template.format(qrels=qrels, run_file=run_path)
        print(f"Running command: {command}")

        try:
            # Execute the evaluation command
            output = subprocess.check_output(command, shell=True, text=True)

            # Extract the evaluation result
            match = pattern.search(output)
            if match:
                result[metric] = match.group(1)
            else:
                result[metric] = "N/A"
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            result[metric] = "Error"
        print("Completed!")

    return result

# Use ThreadPoolExecutor for parallel evaluation
with ThreadPoolExecutor(max_workers=16) as executor:
    # Submit all evaluation tasks
    future_to_file = {executor.submit(evaluate_file, run_file): run_file for run_file in run_files}

    # Retrieve evaluation results
    for future in as_completed(future_to_file):
        run_file = future_to_file[future]
        try:
            result = future.result()
            if result:
                results.append(result)
        except Exception as e:
            print(f"{run_file} generated an exception: {e}")

# Sort results by the file name
results.sort(key=lambda x: x["file"])

# Write evaluation results to a CSV file
with open(output_csv, mode="w", newline='', encoding="utf-8") as csv_file:
    fieldnames = ["file", "dataset_type", "dataset_name"] + list(commands.keys())
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write header
    writer.writeheader()

    # Write each sorted result
    for result in results:
        writer.writerow(result)

print(f"Evaluation results saved to {output_csv}")