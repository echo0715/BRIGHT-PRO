
"""
Qwen embedding searcher implementation.
Uses the Qwen2-7B-Instruct embedding model (alibaba-nlp/gte-qwen2-7b-instruct).
"""

import logging
from typing import Any, Dict, Optional, List
import os.path
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from tqdm import trange

from .base import BaseSearcher

logger = logging.getLogger(__name__)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last token from the hidden states based on attention mask."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def add_instruct_concatenate(texts: List[str], task: str, instruction: str) -> List[str]:
    """Add instruction to texts for better embedding performance."""
    return [f"{instruction} {text}" for text in texts]


def get_scores(query_ids: List[str], doc_ids: List[str], scores: List[List[float]], excluded_ids: set = None) -> Dict[str, Dict[str, float]]:
    """Convert scores to the expected format."""
    if excluded_ids is None:
        excluded_ids = set()
    
    result = {}
    for i, query_id in enumerate(query_ids):
        result[query_id] = {}
        for j, doc_id in enumerate(doc_ids):
            if doc_id not in excluded_ids:
                result[query_id][doc_id] = scores[i][j]
    
    return result


class QwenEmbeddingModel:
    def __init__(self, max_length: int = 8192, device: str = "auto", args=None, **kwargs):
        """
        Initialize the Qwen2 embedding model.
        
        Args:
            max_length: Maximum sequence length
            device: Device to use
            args: Arguments containing cache directories
        """
        self.max_length = kwargs.get('doc_max_length', max_length)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        model_path = 'alibaba-nlp/gte-qwen2-7b-instruct'
        
        # Get cache directory
        cache_dir = getattr(args, 'model_cache_dir', None) if args else None
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_path, 
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        ).eval()
        
        self.batch_size = kwargs.get('encode_batch_size', 64)

    def embed_queries(self, queries: List[str], task: str = None, instruction: str = None) -> List[np.ndarray]:
        """Embed multiple queries."""
        if instruction:
            queries = add_instruct_concatenate(queries, task, instruction)
        
        query_embeddings = []
        for start_idx in trange(0, len(queries), self.batch_size):
            batch_queries = queries[start_idx:start_idx + self.batch_size]
            batch_dict = self.tokenizer(
                batch_queries, 
                max_length=self.max_length, 
                padding=True,
                truncation=True, 
                return_tensors='pt'
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
                query_embeddings.extend([emb.numpy() for emb in embeddings])
        
        return query_embeddings

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed multiple documents with caching."""
        doc_embeddings = []
        
        for start_idx in trange(0, len(documents), self.batch_size):
            batch_docs = documents[start_idx:start_idx + self.batch_size]
            batch_dict = self.tokenizer(
                batch_docs, 
                max_length=self.max_length, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
                doc_embeddings.extend([emb.numpy() for emb in embeddings])
        
        return np.array(doc_embeddings)

    def embed_query(self, query: str, task: str = None, instruction: str = None) -> np.ndarray:
        """Embed a single query."""
        if instruction:
            query = f"{instruction} {query}"
        
        batch_dict = self.tokenizer(
            [query], 
            max_length=self.max_length, 
            padding=True,
            truncation=True, 
            return_tensors='pt'
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
            return embedding[0].numpy()


class QwenSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        """Parse arguments from CLI that you will need to instantiate the searcher."""
        parser.add_argument('--qwen_batch_size', type=int, default=1,
                          help='Batch size for encoding')
        parser.add_argument('--qwen_doc_max_length', type=int, default=None,
                          help='Maximum document length (model-dependent default)')
    
    def __init__(self, args):
        """Initialize the searcher with the arguments."""
        self.args = args
        self.model = None
        self.doc_emb = None
        self.doc_ids = None
        self.documents = None
        
        # Only support qwen2 model, but called as 'qwen'
        batch_size = getattr(args, 'qwen_batch_size', 1)
        doc_max_length = getattr(args, 'qwen_doc_max_length', None)
        
        cache_model_name = 'qwen'
        
        # Load Qwen instructions from configs/qwen/{task}.json when available
        self.instructions: Dict[str, str] = {}
        self.query_instruction = None
        try:
            # Resolve project root relative to this file: searcher/searchers/ -> project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            cfg_path = os.path.join(project_root, 'configs', 'qwen', f'{self.args.task}.json')
            if os.path.isfile(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as jf:
                    cfg = json.load(jf)
                    if 'instructions' in cfg:
                        self.instructions = cfg['instructions']
                        if 'query' in self.instructions and isinstance(self.instructions['query'], str):
                            self.query_instruction = self.instructions['query'].format(task=self.args.task)
        except Exception as e:
            logger.warning(f"Failed to load Qwen instructions for task {getattr(self.args, 'task', '')}: {e}")
        
        # Initialize the embedding model
        kwargs = {
            'encode_batch_size': batch_size,
        }
        if doc_max_length:
            kwargs['doc_max_length'] = doc_max_length
            
        self.model = QwenEmbeddingModel(
            device="auto",
            args=args,
            **kwargs
        )
        
        # Check if documents are already encoded 
        cache_doc_emb_dir = os.path.join(self.args.cache_dir, 'doc_emb', cache_model_name, self.args.task)
        os.makedirs(cache_doc_emb_dir, exist_ok=True)
        cur_cache_file = os.path.join(cache_doc_emb_dir, f'long_context_{batch_size}.npy')
        
        # Load documents from dataset first
        logger.info("Loading documents from dataset...")
        doc_pairs = load_dataset('ya-ir/BRIGHT-PRO', 'documents', cache_dir=self.args.cache_dir)[self.args.task]
        self.doc_ids = []
        self.documents = []
        
        for dp in doc_pairs:
            self.doc_ids.append(dp['id'])
            self.documents.append(dp['content'])
        
        total_docs = len(self.documents)
        logger.info(f"Total documents to encode: {total_docs}")
        
        # Check if cached embeddings exist and if they're complete
        doc_emb = None
        if os.path.isfile(cur_cache_file):
            logger.info(f"Loading cached document embeddings from {cur_cache_file}")
            doc_emb = np.load(cur_cache_file, allow_pickle=True)
            num_cached = doc_emb.shape[0]
            logger.info(f"Found {num_cached} cached embeddings out of {total_docs} documents")
            
            if num_cached >= total_docs:
                logger.info("All documents already encoded")
                self.doc_emb = doc_emb
            else:
                logger.info(f"Resuming encoding from document {num_cached}")
        else:
            logger.info("No cached embeddings found. Starting from scratch.")
        
        # Compute embeddings if needed (either from scratch or resume)
        if doc_emb is None or doc_emb.shape[0] < total_docs:
            start_from = 0 if doc_emb is None else doc_emb.shape[0]
            logger.info(f"Computing document embeddings from index {start_from}...")
            
            for start_idx in trange(start_from, total_docs, batch_size):
                batch_docs = self.documents[start_idx:start_idx + batch_size]
                batch_dict = self.model.tokenizer(
                    batch_docs, 
                    max_length=self.model.max_length, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt'
                ).to(self.model.model.device)
                
                with torch.no_grad():
                    outputs = self.model.model(**batch_dict)
                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
                    
                    if doc_emb is None:
                        doc_emb = embeddings.numpy()
                    else:
                        doc_emb = np.concatenate((doc_emb, embeddings.numpy()), axis=0)
                
                # Save every 1000 iterations
                if (start_idx + batch_size) % 1000 == 0:
                    np.save(cur_cache_file, doc_emb)
                    logger.info(f"Saved {doc_emb.shape[0]} embeddings to cache")
            
            # Final save
            self.doc_emb = doc_emb
            np.save(cur_cache_file, self.doc_emb)
            logger.info(f"Completed encoding. Saved {self.doc_emb.shape[0]} embeddings to {cur_cache_file}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Document embeddings shape: {self.doc_emb.shape}")
        
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search and return results.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of search results with format: {"docid": str, "score": float, "text": str}
        """
        # Embed the query with instruction from config
        query_emb = self.model.embed_query(query, instruction=self.query_instruction)
        query_emb = np.array([query_emb])  # Add batch dimension
        
        # Normalize embeddings
        query_emb = F.normalize(torch.tensor(query_emb), p=2, dim=1).numpy()
        doc_emb_normalized = F.normalize(torch.tensor(self.doc_emb), p=2, dim=1).numpy()
        
        # Compute cosine similarity scores  
        scores = cosine_similarity(query_emb, doc_emb_normalized)
        scores = scores.tolist()[0]  # Remove batch dimension
        
        # Return top-k results using the base class method
        return self.get_top_k_scores(k, self.doc_ids, self.documents, scores)
    
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by ID.
        
        Args:
            docid: Document ID to retrieve
            
        Returns:
            Document dictionary with format: {"docid": str, "text": str} or None if not found
        """
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
        """Return the type of search."""
        return "qwen"

# """
# Qwen2 embedding searcher implementation.
# Follows the same structure and caching conventions as Diver/Inst/Grit searchers.
# Only supports the Qwen2 instruction-tuned embedding model and is registered as
# searcher type `qwen` in the registry.
# """

# import logging
# from typing import Any, Dict, Optional, List
# import os.path
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from datasets import load_dataset

# from .base import BaseSearcher


# logger = logging.getLogger(__name__)


# class QwenEmbeddingModel:
#     def __init__(self, model_path: str = 'alibaba-nlp/gte-qwen2-7b-instruct',
#                  max_length_docs: int = 8192,
#                  max_length_queries: int = 8192,
#                  args=None):
#         """
#         Lightweight wrapper for Qwen2 instruct embedding model using transformers.
#         Uses lazy import to avoid import cost if not used.
#         """
#         from transformers import AutoTokenizer, AutoModel  # type: ignore

#         self.query_instruction = 'Given a web search query, retrieve relevant passages that answer the query'
#         self.doc_instruction = 'Represent this text:'

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_path, trust_remote_code=True, cache_dir=args.model_cache_dir
#         )
#         self.model = AutoModel.from_pretrained(
#             model_path, device_map="auto", trust_remote_code=True, cache_dir=args.model_cache_dir
#         ).eval()

#         self.max_length_docs = max_length_docs
#         self.max_length_queries = max_length_queries

#     @staticmethod
#     def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         last_token_indices = attention_mask.sum(dim=1) - 1
#         batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
#         pooled = last_hidden_state[batch_indices, last_token_indices]
#         return pooled
        
#     def embed_documents(self, documents: List[str], batch_size: int = 1) -> np.ndarray:
#         from tqdm import tqdm
#         all_embeddings: List[np.ndarray] = []
#         for start in tqdm(range(0, len(documents), batch_size), desc="Computing embeddings", unit="batch"):
#             batch_docs = documents[start:start + batch_size]
#             batch_inputs = self.tokenizer(
#                 batch_docs,
#                 max_length=self.max_length_docs,
#                 padding=True,
#                 truncation=True,
#                 return_tensors='pt'
#             ).to(self.model.device)
#             outputs = self.model(**batch_inputs)
#             embeddings = self._last_token_pool(outputs.last_hidden_state, batch_inputs['attention_mask']).cpu().numpy()
#             all_embeddings.append(embeddings)
#         return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 0))

#     def embed_query(self, query: str) -> np.ndarray:
#         inputs = self.tokenizer(
#             [query],
#             max_length=self.max_length_queries,
#             padding=True,
#             truncation=True,
#             return_tensors='pt'
#         ).to(self.model.device)
#         outputs = self.model(**inputs)
#         embedding = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask']).cpu().numpy()[0]
#         return embedding


# class QwenSearcher(BaseSearcher):
#     @classmethod
#     def parse_args(cls, parser):
#         parser.add_argument('--qwen_doc_max_length', type=int, default=8192,
#                             help='Maximum document token length for Qwen2 encoding')
#         parser.add_argument('--qwen_query_max_length', type=int, default=8192,
#                             help='Maximum query token length for Qwen2 encoding')
#         parser.add_argument('--qwen_encode_batch_size', type=int, default=1,
#                             help='Batch size for Qwen2 document encoding')
#         parser.add_argument('--qwen_long_context', type=bool, default=False,
#                             help='Flag retained for cache key parity')

#     def __init__(self, args):
#         self.args = args
#         self.model = None
#         self.doc_emb = None
#         self.doc_ids: List[str] = []
#         self.documents: List[str] = []

#         cache_model_name = 'qwen'

#         doc_max_len = getattr(self.args, 'qwen_doc_max_length', 8192)
#         query_max_len = getattr(self.args, 'qwen_query_max_length', 8192)
#         encode_batch_size = getattr(self.args, 'qwen_encode_batch_size', 1)
#         long_context = getattr(self.args, 'qwen_long_context', False)

#         # Initialize embedding model
#         self.model = QwenEmbeddingModel(
#             model_path='alibaba-nlp/gte-qwen2-7b-instruct',
#             max_length_docs=doc_max_len,
#             max_length_queries=query_max_len,
#             args=self.args,
#         )

#         # Prepare cache path
#         cache_doc_emb_dir = os.path.join(
#             self.args.cache_dir, 'doc_emb', cache_model_name, self.args.task
#         )
#         os.makedirs(cache_doc_emb_dir, exist_ok=True)
#         cur_cache_file = os.path.join(
#             cache_doc_emb_dir, f"long_{long_context}_{encode_batch_size}.npy"
#         )

#         print(cur_cache_file)

#         if os.path.isfile(cur_cache_file):
#             logger.info(f"Loading cached document embeddings from {cur_cache_file}")
#             self.doc_emb = np.load(cur_cache_file, allow_pickle=True)
#             # Load documents/ids
#             print("Loading documents from dataset...")
#             doc_pairs = load_dataset('ya-ir/BRIGHT-PRO', 'documents', cache_dir=self.args.cache_dir)[self.args.task]
#             for dp in doc_pairs:
#                 self.doc_ids.append(dp['id'])
#                 self.documents.append(dp['content'])
#         else:
#             # Load documents and build embeddings
#             print("Loading documents from dataset...")
#             doc_pairs = load_dataset('ya-ir/BRIGHT-PRO', 'documents', cache_dir=self.args.cache_dir)[self.args.task]
#             for dp in doc_pairs:
#                 self.doc_ids.append(dp['id'])
#                 self.documents.append(dp['content'])

#             print(f"Computing embeddings for {len(self.documents)} documents...")
#             with torch.inference_mode():
#                 doc_emb_list = self.model.embed_documents(self.documents, batch_size=encode_batch_size)
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#             self.doc_emb = np.array(doc_emb_list)
#             print(f"Saving embeddings to {cur_cache_file}")
#             np.save(cur_cache_file, self.doc_emb)

#         logger.info("Shape of doc emb %s", str(self.doc_emb.shape))

#     def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
#         query_emb = self.model.embed_query(query)
#         query_emb = np.array([query_emb])

#         # cosine similarity
#         scores = cosine_similarity(query_emb, self.doc_emb)
#         scores = scores.tolist()[0]
#         return self.get_top_k_scores(k, self.doc_ids, self.documents, scores)

#     def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
#         try:
#             idx = self.doc_ids.index(docid)
#             return {
#                 "docid": docid,
#                 "text": self.documents[idx],
#             }
#         except ValueError:
#             return None

#     @property
#     def search_type(self) -> str:
#         return "qwen"

