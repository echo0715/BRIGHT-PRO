import argparse
import json
from typing import Dict, List, Optional, Set, Tuple

from datasets import load_dataset
try:
    from transformers import GPT2TokenizerFast
except Exception as e:
    GPT2TokenizerFast = None  # Will validate at runtime

ALLOWED_TASKS = [
    'biology',
    'earth_science',
    'economics',
    'psychology',
    'robotics',
    'stackoverflow',
    'sustainable_living',
]

def list_tasks(split: str, cache_dir: Optional[str]) -> List[str]:
    dsdict = load_dataset('ya-ir/BRIGHT-PRO', split, cache_dir=cache_dir)
    return sorted(list(dsdict.keys()))


def load_documents(task: str, cache_dir: Optional[str]):
    return load_dataset('ya-ir/BRIGHT-PRO', 'documents', cache_dir=cache_dir)[task]


def load_examples(task: str, cache_dir: Optional[str]):
    return load_dataset('ya-ir/BRIGHT-PRO', 'examples', cache_dir=cache_dir)[task]


def load_aspects(task: str, cache_dir: Optional[str]):
    return load_dataset('ya-ir/BRIGHT-PRO', 'aspects', cache_dir=cache_dir)[task]


def build_doc_to_aspect_map(task: str, cache_dir: Optional[str]) -> Dict[str, Optional[str]]:
    docs = load_documents(task, cache_dir)
    mapping: Dict[str, Optional[str]] = {}
    for doc in docs:
        doc_id = str(doc.get('id'))
        aspect_id = None
        if 'aspect' in doc and doc['aspect'] is not None:
            aspect_id = str(doc['aspect'])
        mapping[doc_id] = aspect_id
    return mapping


_GPT2_TOKENIZER = None


def get_gpt2_tokenizer():
    global _GPT2_TOKENIZER
    if _GPT2_TOKENIZER is None:
        if GPT2TokenizerFast is None:
            raise RuntimeError('transformers not installed; please install transformers to compute GPT-2 token lengths')
        _GPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
        # Avoid warnings about model max length when only counting tokens
        try:
            _GPT2_TOKENIZER.model_max_length = int(1e12)
        except Exception:
            pass
    return _GPT2_TOKENIZER


def compute_document_length_stats(task: str, cache_dir: Optional[str]) -> Tuple[float, float, int]:
    docs = load_documents(task, cache_dir)
    total_chars = 0
    total_tokens = 0
    count = 0
    tokenizer = get_gpt2_tokenizer()
    for doc in docs:
        content = doc.get('content') or ''
        total_chars += len(content)
        total_tokens += len(tokenizer.encode(content, add_special_tokens=False))
        count += 1
    avg_chars = (total_chars / count) if count else 0.0
    avg_tokens = (total_tokens / count) if count else 0.0
    return avg_chars, avg_tokens, count


def compute_positive_doc_stats(task: str, cache_dir: Optional[str], *, long_context: bool) -> Tuple[float, int]:
    examples = load_examples(task, cache_dir)
    field = 'gold_ids_long' if long_context else 'gold_ids'
    total_pos = 0
    num_q = 0
    for ex in examples:
        gold_ids = ex.get(field) or []
        total_pos += len(gold_ids)
        num_q += 1
    avg_pos = (total_pos / num_q) if num_q else 0.0
    return avg_pos, num_q


def compute_avg_unique_aspects_per_query(task: str, cache_dir: Optional[str], *, long_context: bool) -> float:
    doc_to_aspect = build_doc_to_aspect_map(task, cache_dir)
    examples = load_examples(task, cache_dir)
    field = 'gold_ids_long' if long_context else 'gold_ids'
    total_unique = 0
    num_q = 0
    for ex in examples:
        gold_ids = [str(x) for x in (ex.get(field) or [])]
        aspects: Set[str] = set()
        for did in gold_ids:
            aid = doc_to_aspect.get(str(did))
            if aid is not None:
                aspects.add(aid)
        total_unique += len(aspects)
        num_q += 1
    return (total_unique / num_q) if num_q else 0.0


def pick_aspect_text_field(aspects_ds) -> Optional[str]:
    # Prefer likely text/name fields if present; fall back to id length
    candidates = ['name', 'text', 'title', 'description', 'aspect', 'label', 'content', 'topic']
    cols = list(getattr(aspects_ds, 'column_names', []) or [])
    for c in candidates:
        if c in cols:
            return c
    return None


def compute_aspect_string_length_stats(task: str, cache_dir: Optional[str]) -> Tuple[float, float, int, Optional[str]]:
    aspects = load_aspects(task, cache_dir)
    text_field = pick_aspect_text_field(aspects)
    total_len = 0
    total_tokens = 0
    count = 0
    tokenizer = get_gpt2_tokenizer()
    for asp in aspects:
        if text_field is not None:
            s = asp.get(text_field)
        else:
            s = asp.get('id')
        if s is None:
            continue
        s_str = str(s)
        total_len += len(s_str)
        total_tokens += len(tokenizer.encode(s_str, add_special_tokens=False))
        count += 1
    avg_len = (total_len / count) if count else 0.0
    avg_tok = (total_tokens / count) if count else 0.0
    return avg_len, avg_tok, count, text_field


def main():
    parser = argparse.ArgumentParser(description='Compute dataset statistics per task for BRIGHT-PRO.')
    parser.add_argument('--tasks', type=str, default=None, help='Comma-separated task list. Default: all available tasks')
    parser.add_argument('--cache-dir', type=str, default=None, help='HF datasets cache dir')
    parser.add_argument('--long-context', action='store_true', help='Use gold_ids_long for positives/aspects per query')
    parser.add_argument('--save-json', type=str, default=None, help='Optional output path to save all stats as JSON')
    args = parser.parse_args()

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    else:
        # Union of tasks present in examples/documents; will filter to allowed set next
        try:
            tasks = list_tasks('examples', args.cache_dir)
        except Exception:
            tasks = []
        try:
            doc_tasks = list_tasks('documents', args.cache_dir)
            for t in doc_tasks:
                if t not in tasks:
                    tasks.append(t)
        except Exception:
            pass
        tasks.sort()

    # Always restrict to allowed tasks
    tasks = [t for t in tasks if t in ALLOWED_TASKS]

    all_stats: List[Dict[str, object]] = []
    for task in tasks:
        avg_doc_chars, avg_doc_gpt2_tokens, num_docs = compute_document_length_stats(task, args.cache_dir)
        avg_pos_per_query, num_queries = compute_positive_doc_stats(task, args.cache_dir, long_context=args.long_context)
        avg_unique_aspects_per_query = compute_avg_unique_aspects_per_query(task, args.cache_dir, long_context=args.long_context)
        avg_aspect_str_len_chars, avg_aspect_str_len_gpt2_tokens, num_aspects, aspect_text_field = compute_aspect_string_length_stats(task, args.cache_dir)

        stats = {
            'task': task,
            'avg_document_length_chars': avg_doc_chars,
            'avg_document_length_gpt2_tokens': avg_doc_gpt2_tokens,
            'num_documents': num_docs,
            'avg_positive_docs_per_query': avg_pos_per_query,
            'num_queries': num_queries,
            'avg_unique_aspects_per_query': avg_unique_aspects_per_query,
            'avg_aspect_string_length_chars': avg_aspect_str_len_chars,
            'avg_aspect_string_length_gpt2_tokens': avg_aspect_str_len_gpt2_tokens,
            'num_aspects': num_aspects,
            'aspect_text_field_used': aspect_text_field or 'id',
        }

        print(json.dumps(stats, ensure_ascii=False))
        all_stats.append(stats)

    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump({'results': all_stats}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()


