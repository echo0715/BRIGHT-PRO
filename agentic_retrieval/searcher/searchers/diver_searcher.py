"""
Placeholder for custom searcher. Implement your own searcher here.
"""

import logging
from typing import Any, Dict, Optional
import os.path
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
from datasets import load_dataset
from vllm import LLM
from vllm.transformers_utils.tokenizer import get_tokenizer as get_vllm_tokenizer

from .base import BaseSearcher

logger = logging.getLogger(__name__)
class Qwen3EmbeddingModel:
    def __init__(self, model_path, max_length=16384, device="auto", args=None):
        # self.model = LLM(model=model_path, task="embed", gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count())
        self.model = LLM(model=model_path, task="embed", gpu_memory_utilization=0.9, tensor_parallel_size=1, download_dir=args.model_cache_dir)
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
        self.max_length = max_length 
        self.tokenizer = get_vllm_tokenizer(model_path, trust_remote_code=False)

    def truncate_text(self, text):
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > self.max_length:
            text_ids = text_ids[:self.max_length]
            text = self.tokenizer.decode(text_ids)
        return text

    def embed_query(self, query):
        truncated_query = self.truncate_text(query)
        outputs = self.model.embed(truncated_query)
        return outputs[0].outputs.embedding

    def embed_queries(self, query):
        input_queries = ['Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{}'.format(x) for x in query]
        input_queries = [self.truncate_text(x) for x in query]
        outputs = self.model.embed(input_queries)
        return [x.outputs.embedding for x in outputs]

    def embed_doc(self, doc):
        outputs = self.model.embed("Represent this text:{}".format(doc))
        return outputs[0].outputs.embedding

    def embed_docs(self, docs):
        docs = ["Represent this text:{}".format(doc) for doc in docs]
        docs = [self.truncate_text(doc, ) for doc in docs]
        outputs = self.model.embed(docs)
        return [x.outputs.embedding for x in outputs]


class DiverSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        # Parse arguments from CLI that you will need to instantiate the searcher
        pass
    
    def __init__(self, args):
        # Initialize the searcher with the arguments
        self.args = args
        self.searcher = None
        self.model = None
        self.doc_emb = None
        self.doc_ids = None
        self.documents = None
        
        cache_model_name = 'diver'

        model_path = 'AQ-MedAI/Diver-Retriever-4B'
        self.model = Qwen3EmbeddingModel(model_path, max_length=32768, args=self.args)

        # Check if documents are already encoded 
        cache_doc_emb_dir = os.path.join(self.args.cache_dir, 'doc_emb', cache_model_name, self.args.task)
        os.makedirs(cache_doc_emb_dir, exist_ok=True)
        cur_cache_file = os.path.join(cache_doc_emb_dir, f'0.npy')

        if os.path.isfile(cur_cache_file):
            doc_emb = np.load(cur_cache_file,allow_pickle=True)
            self.doc_emb = doc_emb
            
            # Load document IDs and texts if not already loaded
            if self.doc_ids is None or self.documents is None:
                doc_pairs = load_dataset('ya-ir/BRIGHT-PRO', 'documents',cache_dir=self.args.cache_dir)[self.args.task]
                self.doc_ids = []
                self.documents = []
                for dp in doc_pairs:
                    self.doc_ids.append(dp['id'])
                    self.documents.append(dp['content'])
        else:
            doc_pairs = load_dataset('ya-ir/BRIGHT-PRO', 'documents',cache_dir=self.args.cache_dir)[self.args.task]
            self.doc_ids = []
            self.documents = []

            for dp in doc_pairs:
                self.doc_ids.append(dp['id'])
                self.documents.append(dp['content'])
            doc_emb = []
            with torch.inference_mode():
                doc_emb = self.model.embed_docs(self.documents)
            torch.cuda.empty_cache()
            
            # Convert to numpy array and save
            doc_emb = np.array(doc_emb)
            self.doc_emb = doc_emb
            np.save(cur_cache_file, self.doc_emb)
        print("Shape of doc emb", self.doc_emb.shape)
        
        
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        # Implement your search logic here
        # ...
        query_emb = self.model.embed_query(query)
        query_emb = np.array([query_emb])  

        scores = cosine_similarity(query_emb, self.doc_emb)

        scores = scores.tolist()[0]
        return self.get_top_k_scores(k, self.doc_ids, self.documents, scores)
    
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        # Implement your logic for getting a full document by id
        # ...
        return {
            "docid": docid,
            "text": "place-holder-text",
        }
    
    @property
    def search_type(self) -> str:
        return "diver"