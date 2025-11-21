# """
# BGE Reasoner Embed searcher implementation.
# """

# import logging
# from typing import Any, Dict, Optional
# import os.path
# import torch
# import numpy as np
# import torch.nn.functional as F
# from torch import Tensor
# from transformers import AutoTokenizer, AutoModel
# from sklearn.metrics.pairwise import cosine_similarity
# from datasets import load_dataset

# from .base import BaseSearcher

# logger = logging.getLogger(__name__)


# def last_token_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#     if left_padding:
#         return last_hidden_states[:, -1]
#     else:
#         sequence_lengths = attention_mask.sum(dim=1) - 1
#         batch_size = last_hidden_states.shape[0]
#         return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# def get_detailed_instruct(task_description: str, query: str) -> str:
#     return f'Instruct: {task_description}\nQuery: {query}'


# def tokenize_texts(tokenizer, texts, max_length: int, device: str):
#     batch_dict = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=8)
#     batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
#     return batch_dict


# class BGEEmbeddingModel:
#     def __init__(self, model_path="BAAI/bge-reasoner-embed-qwen3-8b-0923", max_length=16000, device="auto", args=None, batch_size=32):
#         self.model_path = model_path
#         self.max_length = max_length
#         self.batch_size = batch_size
#         self.device = device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
        
#         # Initialize tokenizer and model
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=args.cache_dir)
#         self.model = AutoModel.from_pretrained(model_path, cache_dir=args.model_cache_dir)
#         self.model.eval()
#         self.model.to(self.device)
#         self.model.half()
        
#         self.task = 'Given a web search query, retrieve relevant passages that answer the query'

#     def truncate_text(self, text):
#         # Simple truncation based on max_length
#         tokens = self.tokenizer.encode(text, add_special_tokens=False)
#         if len(tokens) > self.max_length:
#             tokens = tokens[:self.max_length]
#             text = self.tokenizer.decode(tokens)
#         return text

#     def embed_query(self, query):
#         truncated_query = self.truncate_text(query)
#         instruct_query = get_detailed_instruct(self.task, truncated_query)
        
#         with torch.no_grad():
#             batch_dict = tokenize_texts(self.tokenizer, [instruct_query], self.max_length, self.device)
#             outputs = self.model(**batch_dict)
#             embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#             embeddings = F.normalize(embeddings, p=2, dim=1)
#             return embeddings[0].cpu().numpy()

#     def embed_queries(self, queries):
#         truncated_queries = [self.truncate_text(q) for q in queries]
#         instruct_queries = [get_detailed_instruct(self.task, q) for q in truncated_queries]
        
#         with torch.no_grad():
#             batch_dict = tokenize_texts(self.tokenizer, instruct_queries, self.max_length, self.device)
#             outputs = self.model(**batch_dict)
#             embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#             embeddings = F.normalize(embeddings, p=2, dim=1)
#             return [emb.cpu().numpy() for emb in embeddings]

#     def embed_doc(self, doc):
#         truncated_doc = self.truncate_text(doc)
        
#         with torch.no_grad():
#             batch_dict = tokenize_texts(self.tokenizer, [truncated_doc], self.max_length, self.device)
#             outputs = self.model(**batch_dict)
#             embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#             embeddings = F.normalize(embeddings, p=2, dim=1)
#             return embeddings[0].cpu().numpy()

#     def embed_docs(self, docs):
#         """Embed documents in batches to reduce memory usage."""
#         all_embeddings = []
        
#         # Process documents in batches
#         for i in range(0, len(docs), self.batch_size):
#             batch_docs = docs[i:i + self.batch_size]
#             truncated_docs = [self.truncate_text(doc) for doc in batch_docs]
            
#             with torch.no_grad():
#                 batch_dict = tokenize_texts(self.tokenizer, truncated_docs, self.max_length, self.device)
#                 outputs = self.model(**batch_dict)
#                 embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#                 embeddings = F.normalize(embeddings, p=2, dim=1)
#                 batch_embeddings = [emb.cpu().numpy() for emb in embeddings]
#                 all_embeddings.extend(batch_embeddings)
                
#                 # Clear GPU cache after each batch to free memory
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
                    
#             print(f"Processed batch {i//self.batch_size + 1}/{(len(docs) + self.batch_size - 1)//self.batch_size}")
        
#         return all_embeddings


# class BGESearcher(BaseSearcher):
#     @classmethod
#     def parse_args(cls, parser):
#         # Parse arguments from CLI that you will need to instantiate the searcher
#         parser.add_argument('--bge_batch_size', type=int, default=32, 
#                            help='Batch size for document embedding to reduce memory usage')
#         pass
    
#     def __init__(self, args):
#         # Initialize the searcher with the arguments
#         self.args = args
#         self.searcher = None
#         self.model = None
#         self.doc_emb = None
#         self.doc_ids = None
#         self.documents = None
        
#         cache_model_name = 'bge_reasoner'

#         # Initialize BGE model with configurable batch size
#         batch_size = getattr(args, 'bge_batch_size', 32)
#         self.model = BGEEmbeddingModel(max_length=16000, args=args, batch_size=batch_size)

#         # Check if documents are already encoded 
#         cache_doc_emb_dir = os.path.join(self.args.cache_dir, 'doc_emb', cache_model_name, self.args.task)
#         os.makedirs(cache_doc_emb_dir, exist_ok=True)
#         cur_cache_file = os.path.join(cache_doc_emb_dir, f'0.npy')

#         if os.path.isfile(cur_cache_file):
#             doc_emb = np.load(cur_cache_file, allow_pickle=True)
#             self.doc_emb = doc_emb
            
#             # Load document IDs and texts if not already loaded
#             if self.doc_ids is None or self.documents is None:
#                 doc_pairs = load_dataset('ya-ir/BRIGHT-PRO', 'documents', cache_dir=self.args.cache_dir)[self.args.task]
#                 self.doc_ids = []
#                 self.documents = []
#                 for dp in doc_pairs:
#                     self.doc_ids.append(dp['id'])
#                     self.documents.append(dp['content'])
#         else:
#             # Load documents from dataset
#             doc_pairs = load_dataset('ya-ir/BRIGHT-PRO', 'documents', cache_dir=self.args.cache_dir)[self.args.task]
#             self.doc_ids = []
#             self.documents = []

#             for dp in doc_pairs:
#                 self.doc_ids.append(dp['id'])
#                 self.documents.append(dp['content'])
            
#             # Compute embeddings
#             doc_emb = []
#             with torch.inference_mode():
#                 doc_emb = self.model.embed_docs(self.documents)
#             torch.cuda.empty_cache()
            
#             # Convert to numpy array and save
#             doc_emb = np.array(doc_emb)
#             self.doc_emb = doc_emb
#             np.save(cur_cache_file, self.doc_emb)
        
#         print("Shape of doc emb", self.doc_emb.shape)
        
        
#     def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
#         # Implement your search logic here
#         query_emb = self.model.embed_query(query)
#         query_emb = np.array([query_emb])  

#         scores = cosine_similarity(query_emb, self.doc_emb)
#         scores = scores.tolist()[0]
#         return self.get_top_k_scores(k, self.doc_ids, self.documents, scores)
    
#     def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
#         # Implement your logic for getting a full document by id
#         # Find the document by ID
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
#         return "bge"


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


class BGESearcher(BaseSearcher):
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
        
        cache_model_name = 'bge_reasoner'

        model_path = 'BAAI/bge-reasoner-embed-qwen3-8b-0923'
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
            # doc_pairs = load_from_disk(f'bright_pro/documents/{self.args.task}')
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