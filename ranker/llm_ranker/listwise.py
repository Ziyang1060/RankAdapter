import copy
import re
from typing import List

from loguru import logger

from ranker import BaseRanker, SearchResult
from ranker.llm_ranker import SYSTEM_PROMPT, SYSTEM_PROMPT_ZH
from utils.LLMClients import create_llm_client
from utils.rank_adapter import RankAdapter
from utils.ranking_utils import single_prompt_chat, CHARACTERS

PROMPT_TEMPLATE = {
    "general": (
        "Query: {query}\n"
        "The following {unit}s are: \n"
        "{passages}"
        "{instruction}rank the {unit}s above based on their relevance to the query.\n"
        '{criteria}'
        "The {unit}s should be listed in descending order using identifiers. "
        "The most relevant {unit}s should be listed first. The output format should be [] > [], e.g., [B] > [C].\n"
        "Output only the ranking results. Do not explain any reason."
    ),
    "general_zh": (
        "查询：{query}\n"
        "以下是{unit}：\n"
        "{passages}"
        "{instruction}根据与查询的相关性对上述{unit}进行排序。\n"
        '{criteria}'
        "你应该以降序方式列出所有{unit}的标识符，并且最相关的{unit}被首先列出。输出格式应为[] > [] > 等，例如，[B] > [C] > [A] > 等。\n"
        "请直接输出排名结果，无需解释原因。"
    )
}

def receive_permutation(ranking, permutation, rank_start, rank_end):
    # clean the response
    # pattern = r"(\[\d+\](?:\s*>\s*\[\d+\])+)"  # match the format of [1] > [2] > [3]
    pattern = r"(\[[A-Z]+\](?:\s*>\s*\[[A-Z]+\])+)"  # match the format of [A] > [B] > [C]
    match = re.findall(pattern, permutation)
    try:
        permutation = match[0].split(">")
    except IndexError:
        logger.warning(f"Rejection: {permutation}")
        return ranking  # return the original ranking if the response is invalid
    for i in range(len(permutation)):
        # permutation[i] = int(permutation[i].strip()[1:-1])
        permutation[i] = permutation[i].strip()
        if permutation[i] not in CHARACTERS:
            permutation[i] = CHARACTERS.index(permutation[i][1:-1][0]) + 1
        else:
            permutation[i] = CHARACTERS.index(permutation[i]) + 1

    # deduplicate
    cut_range_len = rank_end - rank_start
    permutation = [x for x in permutation if 1 <= x <= cut_range_len]

    original_num = len(permutation)
    permutation = list(dict.fromkeys(permutation))  # remove the repeat number but keep the original order
    if len(permutation) < original_num:
        logger.warning(
            f"Repetition: remove the repeat number but keep the original order"
        )

    # add back the missing doc
    missing_strategy = "vanilla"
    if len(permutation) < cut_range_len:
        logger.warning(
            f"Missing: expect {cut_range_len} but got {len(permutation)}, initiating insertion process using "
            f"{missing_strategy} strategy."
        )

        # Find the missing number
        missing_numbers = [i for i in range(1, cut_range_len + 1) if i not in permutation]
        logger.trace(f"Missing numbers identified: {missing_numbers}")

        for missing_idx in missing_numbers:
            logger.debug(
                f"Processing missing item: {ranking[missing_idx - 1].text[:20]}... "
                f"[pos: {missing_idx}, docid: {ranking[missing_idx - 1].docid}]"
            )
            # Add more missing strategy here as needed
            if "vanilla" in missing_strategy:
                logger.info(
                    f"vanilla: No special missing handling strategy selected, appending it to the end"
                )
                permutation.append(missing_idx)

    cut_range = ranking[rank_start:rank_end]
    for j, x in enumerate(permutation):
        ranking[j + rank_start] = cut_range[x-1]

    return ranking


class ListwiseLlmRanker(BaseRanker):
    def __init__(
        self,
        model_name: str,
        window_size: int,
        step_size: int,
        language: str,
        adapter: RankAdapter = None
    ):
        super().__init__(model_name=model_name)
        self.llm_client = create_llm_client(model_name)
        self.window_size = window_size
        self.step_size = step_size
        self.language = language
        self.adapter = adapter

    def compare(self, query: str, docs: List[SearchResult]) -> str:
        self.total_compare += 1
        instruction, query = query.split("_")
        passages = (
                "\n\n".join(
                    [f'{CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)]
                )
                + "\n\n"
        )
        if self.adapter is None:
            user_prompt = PROMPT_TEMPLATE["general" if self.language == "en" else "general_zh"].format(
                query=query, instruction=instruction,
                unit="paragraph" if self.language == "en" else "段落",
                passages=passages,
                criteria=""
            )
        else:
            user_prompt = self.adapter.get_adaptive_ranking_prompt(
                query=query, instruction=instruction, passages=passages,
                prompt_template=PROMPT_TEMPLATE["general" if self.language == "en" else "general_zh"]
            )
        response = single_prompt_chat(
            llm_name=self.model_name,
            llm_client=self.llm_client,
            system_prompt=SYSTEM_PROMPT if self.language == "en" else SYSTEM_PROMPT_ZH,
            user_prompt=user_prompt,
            temperature=0.8
        )

        self.total_prompt_tokens += int(response['usage']['prompt_tokens'])
        self.total_completion_tokens += int(response['usage']['completion_tokens'])
        return response['response']

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        logger.info(f"Reranking {len(ranking)} documents for query: {query}")
        ranking = copy.deepcopy(ranking)
        end_pos = len(ranking)
        start_pos = max(end_pos - self.window_size, 0)
        while start_pos > -self.step_size:
            start_pos = max(start_pos, 0)
            # Before ranking, log the initial docs order
            init_order = ",".join([f"{doc.docid}" for doc in ranking[start_pos: end_pos]])
            logger.trace(f"Initial docs order from {start_pos} to {end_pos}: {init_order}")

            logger.info(f"Reranking {end_pos - start_pos} documents from {start_pos} to {end_pos}")
            result = self.compare(query, ranking[start_pos: end_pos])
            ranking = receive_permutation(ranking, result, start_pos, end_pos)
            end_pos = end_pos - self.step_size
            start_pos = start_pos - self.step_size

        for i, doc in enumerate(ranking):
            doc.score = -i
        return ranking
