import os

import cohere
import numpy as np
import torch
from FlagEmbedding import FlagReranker
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from . import BaseRanker, SearchResult

device = "cuda" if torch.cuda.is_available() else "cpu"

class CrossEncoderRanker(BaseRanker):
    """
    A cross-encoder ranking model that scores and sorts documents based on their similarity to a given query.
    This class uses a cross-encoder transformer model, which directly processes the query-document pairs
    to generate relevance scores.
    """

    def __init__(self, model_name: str, instruction: str="", batch_size: int=8):
        """
        Initializes the CrossEncoderRanker with the specified model, instruction, and batch size.

        Args:
            model_name (str): The name or path of the cross-encoder model to be loaded.
            instruction (str, optional): Instruction text to prepend to queries. Defaults to an empty string.
            batch_size (int, optional): Number of query-document pairs to process in each batch for efficiency. Defaults to 8.
        """
        super().__init__(model_name=model_name)
        self.instruction = instruction
        self.batch_size = batch_size

        # Load the transformer model and tokenizer
        if "rerank-multilingual" in model_name:  # Cohere API
            self.model = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        elif "gte-multilingual-reranker" in model_name:
            model_name = "Alibaba-NLP/gte-multilingual-reranker-base"
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float16,
                # There is a problem with the latest version of the model, so we pin to this revision
                revision="4e88bd5dec38b6b9a7e623755029fc124c319d67"
            ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.eval()
        else:
            self.model = FlagReranker("BAAI/" + model_name, use_fp16=True)

    def rerank(self, query: str, ranking: list[SearchResult]) -> list[SearchResult]:
        query_text = self.instruction + query
        passages = [doc.text for doc in ranking]
        query_passage_pairs = [(query_text, passage) for passage in passages]

        # Handle different model cases for reranking
        if "rerank-multilingual" in self.model_name:  # Cohere API
            response = self.model.rerank(  # noqa
                model=self.model_name,
                query=query,
                documents=passages,
                return_documents=False,
            )
            scores = np.zeros(len(passages))
            for res in response.results:
                scores[res.index] = res.relevance_score

        elif "gte-multilingual-reranker" in self.model_name:
            # Tokenize pairs and create batches for scoring
            inputs = self.tokenizer(query_passage_pairs, padding=True, truncation=True, return_tensors='pt', max_length=8192).to(device)
            dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            scores = []
            with torch.no_grad():
                for input_ids, attention_mask in dataloader:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True) # noqa
                    batch_scores = outputs.logits.view(-1).cpu().numpy()
                    scores.extend(batch_scores)
            scores = np.array(scores)

        else:  # FlagReranker model
            scores = self.model.compute_score(query_passage_pairs, normalize=False)

        # Update scores in SearchResult objects and count comparisons
        for doc, score in zip(ranking, scores):
            doc.score = score
            self.total_compare += 1

        # Sort ranking by score in descending order
        sorted_ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
        return sorted_ranking