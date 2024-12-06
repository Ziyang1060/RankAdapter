from abc import ABC, abstractmethod
from dataclasses import dataclass
import tiktoken


@dataclass
class SearchResult:
    docid: str
    text: str | None
    score: float


class BaseRanker(ABC):
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.truncate_tokenizer = tiktoken.get_encoding("cl100k_base")  # tokenizer used by GPT-4 and GPT-3.5 models

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    @abstractmethod
    def rerank(self, query: str, ranking: list[SearchResult]) -> list[SearchResult]:
        """
        Reranks a list of candidate passages based on their similarity to the input query.

        Args:
            query (str): The input query text for which similar documents are to be ranked.
            ranking (list[SearchResult]): A list of SearchResult objects, each representing a candidate document with a text attribute.

        Returns:
            list[SearchResult]: The list of SearchResult objects, sorted in descending order by similarity score.
        """
        pass

    def truncate(self, text: str, length: int):
        if length == -1:
            return text
        return self.truncate_tokenizer.decode(self.truncate_tokenizer.encode(text)[:length])


from ranker.llm_ranker.pointwise import PointwiseLlmRanker
from ranker.llm_ranker.listwise import ListwiseLlmRanker
from ranker.llm_ranker.setwise import SetwiseLlmRanker
from ranker.bm25 import BM25Ranker
from ranker.bi_encoder import BiEncoderRanker
from ranker.cross_encoder import CrossEncoderRanker
