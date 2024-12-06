#!/bin/bash

# List of ranking methods (写注释，说明白使用方法)
ranking_methods=(
    "bi_encoder"
    "cross_encoder"
    "pointwise"
    "listwise"
    "setwise"
    "bm25"
)

llm_models=(
    "gpt-4o-2024-08-06"
    "deepseek-chat"
    "Qwen2-7B-Instruct"
)

bi_encoder_models=(
    "gte-Qwen2-1.5B-instruct"
    "bge-base-zh-v1.5"
    "bge-large-zh-v1.5"
)

cross_encoder_models=(
    "gte-multilingual-reranker-base"
    "bge-reranker-v2-m3"
    "rerank-multilingual-v3.0"
)

# List of first stage runs
first_stage_runs=(
    # BEIR datasets
    # "run.beir.bm25-flat.trec-covid.txt"
    # "run.beir.bm25-flat.webis-touche2020.txt"

    # Custom dataset
    "run.custom.default.DrRank.txt"
)

# Toggle adapter sub-command (true or false)
use_adapter=true

adapter_modes=(
    "adaptive"
#    "cot"
#    "instruction"
)
adapter_model="Qwen2-7B-Instruct"

# Function to execute ranking
execute_ranking() {
    local model=$1
    local first_stage_run=$2
    local ranking_method=$3
    local adapter_mode=$4

    # Base command
    cmd="python run.py run --first_stage_run \"$first_stage_run\" --save_dir \"data\" --hits 100 --query_length -1 --passage_length -1 --language zh"

    # Add ranking method-specific arguments
    case $ranking_method in
        "bm25")
            cmd+=" bm25 --stopwords_path \"ranker/bm25/stopwords.txt\" --user_dict_path \"ranker/bm25/medical_words.txt\""
            ;;
        "bi_encoder")
            cmd+=" bi_encoder --model_name \"$model\" --batch_size 8"
            ;;
        "cross_encoder")
            cmd+=" cross_encoder --model_name \"$model\" --batch_size 8"
            ;;
        "pointwise")
            cmd+=" pointwise --model_name \"$model\" --method \"score\""
            ;;
        "listwise")
            cmd+=" listwise --model_name \"$model\" --window_size 10 --step_size 5"
            ;;
        "setwise")
            cmd+=" setwise --model_name \"$model\" --num_child 5 --method \"heapsort\" --k 10"
            ;;
    esac

    # Optionally add the adapter sub-command only for llm-based ranking methods
    if [[ "$ranking_method" == "pointwise" || "$ranking_method" == "listwise" || "$ranking_method" == "setwise" ]]; then
        if [ "$use_adapter" = true ]; then
            cmd+=" adapter --model_name \"$adapter_model\" --mode \"$adapter_mode\""
        fi
    fi

    echo "Executing: $cmd"
    eval "$cmd"
}

# Iterate over ranking methods
for ranking_method in "${ranking_methods[@]}"; do
    # Select appropriate model list based on ranking method
    case $ranking_method in
        "bi_encoder")
            model_list=("${bi_encoder_models[@]}")
            ;;
        "cross_encoder")
            model_list=("${cross_encoder_models[@]}")
            ;;
        *)
            # Default to LLM models for pointwise, listwise, setwise, or BM25
            model_list=("${llm_models[@]}")
            ;;
    esac

    # Iterate over models, first stage runs, and adapter modes
    for model in "${model_list[@]}"; do
        for first_stage_run in "${first_stage_runs[@]}"; do
            if [ "$use_adapter" = true ]; then
                for adapter_mode in "${adapter_modes[@]}"; do
                    execute_ranking "$model" "$first_stage_run" "$ranking_method" "$adapter_mode"
                done
            else
                execute_ranking "$model" "$first_stage_run" "$ranking_method"
            fi
        done
    done
done