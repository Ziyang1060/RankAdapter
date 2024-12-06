import re

from ranker import BaseRanker, SearchResult
from ranker.llm_ranker import SYSTEM_PROMPT, SYSTEM_PROMPT_ZH
from utils.LLMClients import create_llm_client
from utils.rank_adapter import RankAdapter
from utils.ranking_utils import single_prompt_chat

PROMPT_TEMPLATE = {
    "score": (
        'Query: {query}\n'
        '{unit}: "{passages}"\n'
        '{instruction}judge the relevance between the query and the {unit} from a scale of 0 to 9.\n'
        '{criteria}'
        'Output only your judgement score. Do not explain any reason.'
    ),
    "score_zh": (
        '查询：{query}\n'
        '{unit}：{passages}\n'
        '{instruction}在分数范围为0-9内判断查询和{unit}之间的相关性。\n'
        '{criteria}'
        '请输出您的判断分数，无需解释原因。'
    ),
}


def extract_numbers(text):
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if len(numbers) == 0:
        numbers.append(0)
    return numbers


class PointwiseLlmRanker(BaseRanker):
    def __init__(
        self,
        model_name: str,
        method: str,
        language: str,
        adapter: RankAdapter = None,
    ):
        super().__init__(model_name=model_name)
        self.llm_client = create_llm_client(model_name)
        self.method = method
        assert method == "score", "Only score method is supported by PointwiseLlmRanker!"
        self.language = language
        self.adapter = adapter

    def rerank(self, query: str, ranking: list[SearchResult]) -> list[SearchResult]:
        instruction, query = query.split("_")
        for doc in ranking:
            self.total_compare += 1
            if self.adapter is None:
                user_prompt = PROMPT_TEMPLATE[self.method if self.language == "en" else self.method + "_zh"].format(
                    query=query, instruction=instruction,
                    unit="paragraph" if self.language == "en" else "段落",
                    passages=doc.text,
                    criteria=""
                )
            else:
                user_prompt = self.adapter.get_adaptive_ranking_prompt(
                    query=query, instruction=instruction, passages=doc.text.strip(),
                    prompt_template=PROMPT_TEMPLATE[self.method if self.language == "en" else self.method + "_zh"]
                )
            response = single_prompt_chat(
                llm_name=self.model_name,
                llm_client=self.llm_client,
                system_prompt=SYSTEM_PROMPT if self.language == "en" else SYSTEM_PROMPT_ZH,
                user_prompt=user_prompt,
                temperature=0.0
            )
            doc.score = float(extract_numbers(response["response"])[0])
            self.total_completion_tokens += int(response["usage"]["completion_tokens"])
            self.total_prompt_tokens += int(response["usage"]["prompt_tokens"])

        ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
        return ranking
