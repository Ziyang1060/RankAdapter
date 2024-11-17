import torch
from FlagEmbedding import FlagModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from . import BaseRanker, SearchResult

device = "cuda" if torch.cuda.is_available() else "cpu"


class BiEncoderRanker(BaseRanker):
    """
    A bi-encoder ranking model that scores and sorts documents based on their similarity to a given query.
    This class supports two models: `SentenceTransformer` and `FlagModel`, with flexible initialization
    based on the model name provided.
    """

    def __init__(self, model_name: str, instruction: str='', batch_size: int=8):
        """
        Initializes the BiEncoderRanker with the specified model, instruction, and batch size.

        Args:
            model_name (str): The name or path of the bi-encoder model to be loaded.
            instruction (str, optional): Instruction text to prepend to queries. Defaults to an empty string.
            batch_size (int, optional): Number of passages to encode in each batch for efficient processing. Defaults to 8.
        """
        super().__init__(model_name=model_name)
        self.instruction = instruction
        self.batch_size = batch_size

        # Load the appropriate model based on model name
        if "gte-Qwen2-1.5B-instruct" in model_name:
            self.model = SentenceTransformer("Alibaba-NLP/" + model_name, trust_remote_code=True, device=device)
            self.model.eval()
            self.model.max_seq_length = 8192
        else:
            self.model = FlagModel(
                "BAAI/" + model_name,
                normalize_embeddings=True,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=True,
            )  # Enable use_fp16 for improved performance with slight precision degradation.

    def rerank(self, query: str, ranking: list[SearchResult]) -> list[SearchResult]:
        # Process query with instruction if specified
        if "gte-Qwen2-1.5B-instruct" in self.model_name and self.instruction != "":
            query_embedding = self.model.encode(query, prompt=f"Instruct: {self.instruction}\nQuery: ")
        else:
            query = self.instruction + query
            query_embedding = self.model.encode(query)

        # Encode each passage in batches and store embeddings
        passages = [doc.text for doc in ranking]
        passage_embeddings = []
        for i in range(0, len(passages), self.batch_size):
            batch_embeddings = self.model.encode(passages[i:i + self.batch_size])
            passage_embeddings.extend(batch_embeddings)

        # Compute cosine similarity between query and passage embeddings
        scores = cosine_similarity([query_embedding], passage_embeddings).flatten()
        for doc, score in zip(ranking, scores):
            doc.score = score
            self.total_compare += 1  # Increment comparison counter

        # Sort ranking in descending order by score
        sorted_ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
        return sorted_ranking
