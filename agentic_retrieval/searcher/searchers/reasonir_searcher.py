"""
ReasonIR embed searcher implementation.
Follows the same structure and caching conventions as Diver and BGE searchers.
"""

import logging
from typing import Any, Dict, Optional
import os.path
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModel

from .base import BaseSearcher

logger = logging.getLogger(__name__)


class ReasonIREmbeddingModel:
    def __init__(self, model_path: str = 'reasonir/ReasonIR-8B', max_length: int = 32768, args=None, instructions: Optional[Dict[str, str]] = None):
        self.model_path = model_path
        self.max_length = max_length
    
        cache_dir = getattr(args, 'model_cache_dir', None) if args else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, cache_dir=cache_dir, torch_dtype="auto"
        )
        self.model.eval()
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        task = args.task if args else None
        # Default instructions, overridden by config if provided
        self.query_instruction = ''
        self.doc_instruction = ''

        # Override from config-provided instructions
        if instructions and task:
            if 'query' in instructions and isinstance(instructions['query'], str):
                self.query_instruction = instructions['query'].format(task=task)
            if 'document' in instructions and isinstance(instructions['document'], str):
                self.doc_instruction = instructions['document']

    def _truncate(self, text: str) -> str:
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            text = self.tokenizer.decode(input_ids)
        return text

    def embed_query(self, query: str) -> np.ndarray:
        text = self._truncate(query)
        with torch.no_grad():
            emb = self.model.encode([text], instruction=self.query_instruction, batch_size=1, max_length=self.max_length)
        # encode returns list-like np array
        return np.array(emb[0])

    def embed_queries(self, queries):
        queries = [self._truncate(q) for q in queries]
        with torch.no_grad():
            embs = self.model.encode(queries, instruction=self.query_instruction, batch_size=1, max_length=self.max_length)
        return [np.array(e) for e in embs]

    def embed_doc(self, doc: str) -> np.ndarray:
        text = self._truncate(doc)
        with torch.no_grad():
            emb = self.model.encode([text], instruction=self.doc_instruction, batch_size=1, max_length=self.max_length)
        return np.array(emb[0])

    def embed_docs(self, docs):
        docs = [self._truncate(d) for d in docs]
        with torch.no_grad():
            embs = self.model.encode(docs, instruction=self.doc_instruction, batch_size=1, max_length=self.max_length)
        return [np.array(e) for e in embs]


class ReasonIRSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        # Keep interface parity with other searchers
        pass
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.doc_emb = None
        self.doc_ids = None
        self.documents = None

        cache_model_name = 'reasonir'

        # Load ReasonIR instructions from configs/reasonir/{task}.json when available
        instructions: Dict[str, str] = {}
        try:
            # Resolve project root relative to this file: searcher/searchers/ -> project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            cfg_path = os.path.join(project_root, 'configs', 'reasonir', f'{self.args.task}.json')
            if os.path.isfile(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as jf:
                    cfg = json.load(jf)
                    instructions = cfg['instructions']
        except Exception as e:
            logger.warning(f"Failed to load ReasonIR instructions for task {getattr(self.args, 'task', '')}: {e}")

        model_path = 'reasonir/ReasonIR-8B'
        self.model = ReasonIREmbeddingModel(
            model_path=model_path,
            max_length=32768,
            args=self.args,
            instructions=instructions,
        )

        # Prepare cache dir and file
        cache_doc_emb_dir = os.path.join(self.args.cache_dir, 'doc_emb', cache_model_name, self.args.task)
        os.makedirs(cache_doc_emb_dir, exist_ok=True)
        cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

        if os.path.isfile(cur_cache_file):
            doc_emb = np.load(cur_cache_file, allow_pickle=True)
            self.doc_emb = doc_emb

            # Load doc ids and documents if not already
            if self.doc_ids is None or self.documents is None:
                doc_pairs = load_dataset('ya-ir/BRIGHT-PRO', 'documents', cache_dir=self.args.cache_dir)[self.args.task]
                self.doc_ids = []
                self.documents = []
                for dp in doc_pairs:
                    self.doc_ids.append(dp['id'])
                    self.documents.append(dp['content'])
        else:
            # Load docs and compute embeddings
            doc_pairs = load_dataset('ya-ir/BRIGHT-PRO', 'documents', cache_dir=self.args.cache_dir)[self.args.task]
            self.doc_ids = []
            self.documents = []

            for dp in doc_pairs:
                self.doc_ids.append(dp['id'])
                self.documents.append(dp['content'])

            with torch.inference_mode():
                doc_emb_list = self.model.embed_docs(self.documents)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            doc_emb = np.array(doc_emb_list)
            self.doc_emb = doc_emb
            np.save(cur_cache_file, self.doc_emb)

        print("Shape of doc emb", self.doc_emb.shape)
    
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:

        query_emb = self.model.embed_query(query)
        query_emb = np.array([query_emb])  

        scores = cosine_similarity(query_emb, self.doc_emb)

        scores = scores.tolist()[0]

        return self.get_top_k_scores(k, self.doc_ids, self.documents, scores)
    
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        return {
            "docid": docid,
            "text": "place-holder-text",
        }
    
    @property
    def search_type(self) -> str:
        return "reasonir"


