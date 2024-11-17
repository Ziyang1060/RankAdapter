import jieba
from rank_bm25 import BM25Okapi

from .. import BaseRanker, SearchResult


def _load_stopwords(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        stopwords = set(file.read().strip().split("\n"))
    return stopwords


class BM25Ranker(BaseRanker):
    """
    A BM25-based ranking model that scores and ranks documents based on their relevance to a given query.
    This class uses the BM25Okapi model and Jieba for text preprocessing.
    """

    def __init__(self, stopwords_path="ranker/bm25/stopwords.txt", user_dict_path="ranker/bm25/medical_words.txt"):
        """
        Initializes the BM25Ranker with custom stopwords and user-defined Jieba dictionary.

        Args:
            stopwords_path (str): The path to the file containing stopwords.
            user_dict_path (str): The path to the file containing custom dictionary words for Jieba.
        """
        super().__init__()
        jieba.load_userdict(user_dict_path)
        self.stopwords = _load_stopwords(stopwords_path)
        self.bm25 = None

    def preprocess_text(self, text):
        """
        Tokenizes the text using Jieba and removes stopwords.
        """
        words = jieba.lcut(text)
        words = [word for word in words if word not in self.stopwords]
        return words

    def compute_scores(self, query, documents):
        """
        Computes BM25 relevance scores for the query against a list of documents.

        Args:
            query (str): The input query text.
            documents (list[str]): A list of documents to rank against the query.

        Returns:
            list[float]: A list of relevance scores, one per document.
        """
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        self.bm25 = BM25Okapi(processed_docs)
        processed_query = self.preprocess_text(query)
        scores = self.bm25.get_scores(processed_query)
        return scores

    def rerank(self, query: str, ranking: list[SearchResult]) -> list[SearchResult]:
        documents = [doc.text for doc in ranking]

        processed_docs = [self.preprocess_text(doc) for doc in documents]
        self.bm25 = BM25Okapi(processed_docs)
        processed_query = self.preprocess_text(query)
        scores = self.bm25.get_scores(processed_query)

        for doc, score in zip(ranking, scores):
            doc.score = score

        # Sort ranking by score in descending order
        sorted_ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
        return sorted_ranking


if __name__ == "__main__":
    # Test the BM25Ranker class with sample data
    drs = [
        "周俭，男，1967年出生，国家杰出青年科学基金获得者，教育部长江学者特聘教授，上海市领军人才...",
        "王鲁，主任医师，副教授，肝脏外科主任，复旦大学附属肿瘤医院肝脏外科...",
        "于占江，男，肿瘤外科病房副主任、齐齐哈尔医学院附属第三医院副主任医师、副教授...",
        "黄洁夫，男，中国医学科学院北京协和医院主任医师，教授，原肝脏外科主任...",
    ]
    query = "擅长使用手术治疗肝癌的医生专家"

    bm25_ranker = BM25Ranker()
    scores = bm25_ranker.compute_scores(query=query, documents=drs)
    print(scores)