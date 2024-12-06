import argparse
import json
import os.path
import random
import sys
import time

from loguru import logger
from pyserini.search import get_topics, LuceneSearcher
from tqdm import tqdm

from utils.ranking_utils import INSTRUCTIONS

from ranker import (
    BM25Ranker,
    BiEncoderRanker,
    CrossEncoderRanker,
    ListwiseLlmRanker,
    PointwiseLlmRanker,
    SetwiseLlmRanker,
    SearchResult,
)
from utils.rank_adapter import RankAdapter

random.seed(929)


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def write_run_file(path, results, tag):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    logger.info(f'Writing run file to {path}.')
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def main(args):
    adapter_identifier = ""
    adapter = None

    if args.adapter:
        adapter = RankAdapter(
            llm=args.adapter.model_name,
            mode=args.adapter.mode,
            language=args.run.language,
        )
        adapter_identifier = f"_adapter_{args.adapter.model_name}_{args.adapter.mode}"

    if args.bm25:
        run_identifier = "bm25"
        ranker = BM25Ranker(
            stopwords_path=args.bm25.stopwords_path,
            user_dict_path=args.bm25.user_dict_path,
        )
    elif args.bi_encoder:
        run_identifier = f"bi-encoder_{args.bi_encoder.model_name}_{args.bi_encoder.batch_size}"
        ranker = BiEncoderRanker(
            model_name=args.bi_encoder.model_name,
            batch_size=args.bi_encoder.batch_size,
        )
    elif args.cross_encoder:
        run_identifier = f"cross-encoder_{args.cross_encoder.model_name}_{args.cross_encoder.batch_size}"
        ranker = CrossEncoderRanker(
            model_name=args.cross_encoder.model_name,
            batch_size=args.cross_encoder.batch_size,
        )
    elif args.pointwise:
        run_identifier = f"pointwise_{args.pointwise.model_name}_{args.pointwise.method}"
        ranker = PointwiseLlmRanker(
            model_name=args.pointwise.model_name,
            method=args.pointwise.method,
            language=args.run.language,
            adapter=adapter
        )
    elif args.setwise:
        run_identifier = f"setwise_{args.setwise.model_name}_{args.setwise.method}_{args.setwise.num_child}_{args.setwise.k}"
        ranker = SetwiseLlmRanker(
            model_name=args.setwise.model_name,
            method=args.setwise.method,
            num_child=args.setwise.num_child,
            k=args.setwise.k,
            language=args.run.language,
            adapter=adapter
        )
    elif args.listwise:
        run_identifier = f"listwise_{args.listwise.model_name}_{args.listwise.window_size}_{args.listwise.step_size}"
        ranker = ListwiseLlmRanker(
            model_name=args.listwise.model_name,
            window_size=args.listwise.window_size,
            step_size=args.listwise.step_size,
            language=args.run.language,
            adapter=adapter
        )
    else:
        raise ValueError("Illegal ranker specified.")

    # Set the run file name and log file name
    parts = args.run.first_stage_run.split('.')
    parts[2] = run_identifier + adapter_identifier
    run_save_path = os.path.join(args.run.save_dir, 'run', '.'.join(parts[:-1]))
    log_save_path = os.path.join(args.run.save_dir, 'log', '.'.join(parts[:-1]))
    logger.add(log_save_path + '.log', level='TRACE')
    logger.debug("Run file will be saved to: " + run_save_path + '.txt')
    logger.debug("Log file will be saved to: " + log_save_path + '.log')

    query_map = {}
    dataset_name = args.run.first_stage_run.split('.')[-2]
    if args.run.first_stage_run.split('.')[1] == 'custom':
        if os.path.exists(f'data/custom_dataset/{dataset_name}'):
            docstore = LuceneSearcher(f'data/custom_dataset/{dataset_name}/index')
            query_file = f'data/custom_dataset/{dataset_name}/queries.tsv'
        else:
            raise ValueError(f'Custom dataset {dataset_name} not found.')

        with open(query_file, mode='r', encoding='utf-8') as file:
            for line in file:
                qid, content = line.strip().split('\t')
                query_map[str(qid)] = content
    else:
        index_mapping = {
            'trec-covid': 'beir-v1.0.0-trec-covid.flat',
            'webis-touche2020': 'beir-v1.0.0-webis-touche2020.flat',
        }
        docstore = LuceneSearcher.from_prebuilt_index(index_mapping[dataset_name])

        topics_mapping = {
            'trec-covid': 'beir-v1.0.0-trec-covid-new_trec',
            'webis-touche2020': 'beir-v1.0.0-webis-touche2020-new_trec',
        }
        topics = get_topics(topics_mapping[dataset_name])
        for topic_id in list(topics.keys()):
            text = topics[topic_id]['title']
            query_map[str(topic_id)] = ranker.truncate(text, args.run.query_length)
    logger.info(f'Loaded {len(query_map)} queries. Dataset: {dataset_name}')

    run_load_path = os.path.join(args.run.save_dir, 'first_stage_run', args.run.first_stage_run)
    logger.info(f'Loading first stage run from {run_load_path}.')
    first_stage_rankings = []
    with open(run_load_path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map[current_qid],
                                                 current_ranking if args.run.hits== -1 else current_ranking[:args.run.hits]))
                current_ranking = []
                current_qid = qid
            if len(current_ranking) >= args.run.hits:
                continue
            data = json.loads(docstore.doc(docid).raw())
            try:
                text = data['text']
                if 'title' in data:
                    text = f'Title: {data["title"]} Content: {text}'
            except KeyError:
                text = data['contents']
            text = ranker.truncate(text, args.run.passage_length)
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        first_stage_rankings.append((current_qid, query_map[current_qid],
                                     current_ranking if args.run.hits== -1 else current_ranking[:args.run.hits]))

    reranked_results = []
    tic = time.time()
    for qid, query, ranking in tqdm(first_stage_rankings):
        if args.run.shuffle_ranking is not None:
            if args.run.shuffle_ranking == 'random':
                random.shuffle(ranking)
            elif args.run.shuffle_ranking == 'inverse':
                ranking = ranking[::-1]
            else:
                raise ValueError(f'Invalid shuffle ranking method: {args.run.shuffle_ranking}.')
        reranked_results.append((qid, query, ranker.rerank(INSTRUCTIONS[dataset_name] + "_" + query, ranking)))
    toc = time.time()

    logger.info(f'Avg comparisons: {ranker.total_compare / len(reranked_results):.2f}')
    logger.info(f'Avg prompt tokens: {ranker.total_prompt_tokens / len(reranked_results):.2f}')
    logger.info(f'Avg completion tokens: {ranker.total_completion_tokens / len(reranked_results):.2f}')
    logger.info(f'Avg time per query: {(toc - tic) / len(reranked_results):.2f}')
    if adapter is not None:
        logger.info(f"Avg prompt tokens consumed by RankAdapter: "
                    f"{adapter.total_prompt_tokens / len(reranked_results):.2f}")
        logger.info(f"Avg completion tokens consumed by RankAdapter: "
                    f"{adapter.total_completion_tokens / len(reranked_results):.2f}")

    write_run_file(run_save_path + '.txt', reranked_results, 'RankAdapter')
    logger.info(f'Done! {len(reranked_results)} queries reranked.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')

    run_parser = commands.add_parser('run')
    run_parser.add_argument(
        '--first_stage_run', type=str,
        help='File name of the first stage first_stage_run file (TREC format) to rerank.')
    run_parser.add_argument('--save_dir', type=str,
                            help='Directory to save the reranked run file (TREC format) and log file.', default='data')
    run_parser.add_argument('--hits', type=int, default=-1, help='Number of hits to rerank.')
    run_parser.add_argument('--query_length', type=int, default=-1,
                            help='Max length of the query. Default to -1, which means no truncation.')
    run_parser.add_argument('--passage_length', type=int, default=-1,
                            help='Max length of the passage. Default to -1, which means no truncation.')
    run_parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    run_parser.add_argument('--language', type=str, default='en', choices=['zh', 'en'])

    bm25_parser = commands.add_parser('bm25')
    bm25_parser.add_argument('--stopwords_path', help='Path to the file containing stopwords.')
    bm25_parser.add_argument('--user_dict_path', help='Path to the file containing custom dictionary words for Jieba.')

    bi_encoder_parser = commands.add_parser('bi_encoder')
    bi_encoder_parser.add_argument('--model_name', type=str,
                                   help='The name or path of the bi-encoder model to be loaded.')
    bi_encoder_parser.add_argument('--batch_size', type=int, default=8,
                                   help='Number of passages to encode in each batch for efficient processing.')

    cross_encoder_parser = commands.add_parser('cross_encoder')
    cross_encoder_parser.add_argument('--model_name', type=str,
                                      help='The name or path of the cross-encoder model to be loaded.')
    cross_encoder_parser.add_argument('--batch_size', type=int, default=8,
                                      help='Number of passages to encode in each batch for efficient processing.')

    adapter_parser = commands.add_parser('adapter', help='Generate adaptive ranking criteria using RankAdapter.')
    adapter_parser.add_argument('--model_name', type=str,
                                help='Name of the LLM to use for rank rule adaptation.')
    adapter_parser.add_argument('--mode', type=str, default='general',
                                choices=["cot", "instruction", "adaptive"])

    pointwise_parser = commands.add_parser('pointwise')
    pointwise_parser.add_argument('--model_name', type=str, help='Name of the LLM to use for reranking.')
    pointwise_parser.add_argument('--method', type=str, default='score',
                                  choices=['qlm', 'yes_no', 'score'])

    setwise_parser = commands.add_parser('setwise')
    setwise_parser.add_argument('--model_name', type=str, help='Name of the LLM to use for reranking.')
    setwise_parser.add_argument('--num_child', type=int, default=3)
    setwise_parser.add_argument('--method', type=str, default='heapsort', choices=['heapsort', 'bubblesort'])
    setwise_parser.add_argument('--k', type=int, default=10)

    listwise_parser = commands.add_parser('listwise')
    listwise_parser.add_argument('--model_name', type=str, help='Name of the LLM to use for reranking.')
    listwise_parser.add_argument('--window_size', type=int, default=3)
    listwise_parser.add_argument('--step_size', type=int, default=1)

    args = parse_args(parser, commands)

    # Integrity check
    arg_dict = vars(args)

    if 'run' not in arg_dict or arg_dict['run'] is None:
        raise ValueError("The 'run' sub-command must always be specified.")

    reranking_methods = ['pointwise', 'listwise', 'setwise', 'bi_encoder', 'cross_encoder', 'bm25']
    selected_methods = [method for method in reranking_methods if arg_dict.get(method) is not None]
    if len(selected_methods) != 1:
        raise ValueError(
            "Exactly one of --pointwise, --listwise, --setwise, --bi_encoder, --cross_encoder, or --bm25 must be specified."
        )

    if arg_dict.get('adapter') is not None and not any(
            arg_dict.get(method) is not None for method in ['pointwise', 'listwise', 'setwise']
    ):
        raise ValueError(
            "The 'adapter' sub-command can only be used with one of --pointwise, --listwise, or --setwise."
        )

    main(args)
