"""
BM25 searcher implementation using pyserini analyzer and gensim's LuceneBM25Model.
Follows the same structure and dataset-loading conventions as other searchers.
"""

import logging
from typing import Any, Dict, Optional
import os.path

import numpy as np
from datasets import load_dataset

from .base import BaseSearcher


logger = logging.getLogger(__name__)


class BM25Searcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument('--bm25_k1', type=float, default=0.9, help='BM25 k1 parameter')
        parser.add_argument('--bm25_b', type=float, default=0.4, help='BM25 b parameter')
        # relies on common args: --cache_dir, --model_cache_dir, --task
        return
    
    def __init__(self, args):
        self.args = args
        self.doc_ids = None
        self.documents = None
        self.dictionary = None
        self.model = None
        self.bm25_index = None
        self.analyzer = None

        # Load documents (align with other searchers)
        doc_pairs = load_dataset('ya-ir/BRIGHT-PRO', 'documents', cache_dir=self.args.cache_dir)[self.args.task]
        self.doc_ids = []
        self.documents = []
        for dp in doc_pairs:
            self.doc_ids.append(dp['id'])
            self.documents.append(dp['content'])

        # Build BM25 components
        from pyserini import analysis
        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model
        from gensim.similarities import SparseMatrixSimilarity

        self.analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
        corpus_tokens = [self.analyzer.analyze(x) for x in self.documents]
        self.dictionary = Dictionary(corpus_tokens)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=getattr(args, 'bm25_k1', 0.9), b=getattr(args, 'bm25_b', 0.4))
        bm25_corpus = self.model[list(map(self.dictionary.doc2bow, corpus_tokens))]
        self.bm25_index = SparseMatrixSimilarity(
            bm25_corpus,
            num_docs=len(corpus_tokens),
            num_terms=len(self.dictionary),
            normalize_queries=False,
            normalize_documents=False,
        )
        logger.info("BM25 index built for %d documents", len(self.documents))
        
    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        if not query:
            return []
        query_tokens = self.analyzer.analyze(query)
        bm25_query = self.model[self.dictionary.doc2bow(query_tokens)]
        scores = self.bm25_index[bm25_query].tolist()
        return self.get_top_k_scores(k, self.doc_ids, self.documents, scores)
    
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        try:
            idx = self.doc_ids.index(docid)
            return {
                "docid": docid,
                "text": self.documents[idx],
            }
        except ValueError:
            return None
    
    @property
    def search_type(self) -> str:
        return "bm25"
    
    def search_description(self, k: int = 10) -> str:
        return f"Perform a BM25 search on the document collection. Returns top-{k} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."
    
    def get_document_description(self) -> str:
        return "Retrieve a full document by its docid."