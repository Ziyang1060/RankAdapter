import copy

from loguru import logger

from ranker import BaseRanker, SearchResult
from ranker.llm_ranker import SYSTEM_PROMPT
from utils.LLMClients import create_llm_client
from utils.rank_adapter import RankAdapter
from utils.ranking_utils import single_prompt_chat, CHARACTERS

PROMPT_TEMPLATE = {
    "general": (
        "Query: {query}\n"
        "The following {unit}s are: \n"
        "{passages}"
        "{instruction}which {unit} above is most relevant to the query?\n"
        "{criteria}"
        "Output only the identifier of the most relevant {unit}. Do not explain any reason.\n"
    ),
    "general_zh": (
        "查询：{query}\n"
        "以下是{unit}：\n"
        "{passages}"
        "{instruction}哪个{unit}最相关于查询？\n"
        "{criteria}"
        "只输出最相关的{unit}的标识符。不要解释任何原因。\n"
    ),
}


def fix_response(response, query=""):
    raw_response = response
    response = response.split(":")[-1].strip()
    if response not in CHARACTERS:
        response = response.split("\n")[0]
        if response.strip() not in CHARACTERS:
            response = response.split(", ")[0]
            if response.strip() not in CHARACTERS:
                logger.error(f"[query: {query}] Unexpected output: {raw_response}, return A")
                response = "A"
    return response


class SetwiseLlmRanker(BaseRanker):
    def __init__(
        self,
        model_name: str,
        num_child: int,
        k: int,
        method: str,
        language: str,
        adapter: RankAdapter = None,
    ):
        super().__init__(model_name=model_name)
        self.llm_client = create_llm_client(model_name)
        self.num_child = num_child
        self.k = k
        self.method = method
        self.language = language
        self.adapter = adapter

    def compare(self, query: str, docs: list):
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
            system_prompt=SYSTEM_PROMPT if self.language == "en" else SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )

        output = response["response"]
        self.total_prompt_tokens += int(response["usage"]["prompt_tokens"])
        self.total_completion_tokens += int(response["usage"]["completion_tokens"])

        output = fix_response(output, query=query)
        return output

    def heapify(self, arr, n, i, query):
        # Find largest among root and children
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[
                self.num_child * i + 1 : min((self.num_child * (i + 1) + 1), n)
            ]
            inds = [i] + list(
                range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n))
            )
            output = self.compare(query, docs)
            try:
                best_ind = CHARACTERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heapSort(self, arr, query, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0, query)

    def rerank(self, query: str, ranking: list[SearchResult]) -> list[SearchResult]:
        original_ranking = copy.deepcopy(ranking)

        if self.method == "heapsort":
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))
        elif self.method == "bubblesort":
            last_start = len(ranking) - (self.num_child + 1)

            for i in range(self.k):
                start_ind = last_start
                end_ind = last_start + (self.num_child + 1)
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    output = self.compare(query, ranking[start_ind:end_ind])
                    try:
                        best_ind = CHARACTERS.index(output)
                    except ValueError:
                        best_ind = 0
                    if best_ind != 0:
                        ranking[start_ind], ranking[start_ind + best_ind] = (
                            ranking[start_ind + best_ind],
                            ranking[start_ind],
                        )
                        if not is_change:
                            is_change = True
                            if (
                                last_start != len(ranking) - (self.num_child + 1)
                                and best_ind == len(ranking[start_ind:end_ind]) - 1
                            ):
                                last_start += len(ranking[start_ind:end_ind]) - 1

                    if start_ind == i:
                        break

                    if not is_change:
                        last_start -= self.num_child

                    start_ind -= self.num_child
                    end_ind -= self.num_child

        else:
            raise NotImplementedError(f"Method {self.method} is not implemented.")

        results = []
        top_doc_ids = set()
        rank = 1

        for i, doc in enumerate(ranking[: self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results
