import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from datasets import load_dataset

# Optional dependency for Excel export
try:
	import pandas as pd
except ImportError:
	pd = None


def parse_score_entry(entry):
	"""
	Normalize a score.json entry to a ranked list of document ids (strings).

	The repository uses two possible formats in different places:
	- Mapping: {doc_id: score, ...} per query id
	- Ranked list: [doc_id, doc_id, ...] per query id

	This function returns a list of doc ids sorted from high to low score
	when a mapping is given, or the list as-is when already ranked.
	"""
	if isinstance(entry, dict):
		# Sort by score descending, keep only doc ids
		return [doc_id for doc_id, _ in sorted(entry.items(), key=lambda x: x[1], reverse=True)]
	elif isinstance(entry, list):
		# Already a ranked list
		return entry
	else:
		raise ValueError(f"Unsupported score entry type: {type(entry)}")


def build_aspect_maps(task: str, cache_dir: str) -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
	"""
	Build maps:
	- doc_id_to_aspect_id: map from document id (string) to its aspect id (string)
	- aspect_id_to_weight: map from aspect id (string) to its weight (float)

	Notes:
	- Documents split has fields: id (string), content (string), aspect (string)
	- Aspects split has fields: id (string), weight (float)
	"""
	documents = load_dataset('ya-ir/BRIGHT-PRO-WITH-ASPECT', 'documents', cache_dir=cache_dir)[task]
	aspects = load_dataset('ya-ir/BRIGHT-PRO-WITH-ASPECT', 'aspects', cache_dir=cache_dir)[task]

	doc_id_to_aspect_id: Dict[str, Optional[str]] = {}
	for doc in documents:
		# Ensure string ids
		doc_id_to_aspect_id[str(doc['id'])] = str(doc['aspect']) if 'aspect' in doc and doc['aspect'] is not None else None

	aspect_id_to_weight: Dict[str, float] = {}
	for asp in aspects:
		aspect_id_to_weight[str(asp['id'])] = float(asp['weight'])

	return doc_id_to_aspect_id, aspect_id_to_weight


def compute_weighted_aspect_recall_at_k(
	ranked_doc_ids: List[str],
	gold_doc_ids: List[str],
	doc_id_to_aspect_id: Dict[str, Optional[str]],
	aspect_id_to_weight: Dict[str, float],
	k: int,
) -> float:
	"""
	Compute Weighted Aspect Recall@k for a single query.

	Definition (normalized): A-Recall@k = sum_{a in A_q} w_a * 1{C_a(k) >= 1} / sum_{a in A_q} w_a,
	where A_q are the aspects present among the gold documents for this query, and C_a(k)
	is the count of retrieved documents within the top-k positions whose aspect is a
	(credit each aspect at most once).
	"""
	# Identify the set of aspects relevant to this query (from gold documents)
	gold_aspects = []
	for doc_id in gold_doc_ids:
		asp = doc_id_to_aspect_id.get(str(doc_id))
		if asp is not None:
			gold_aspects.append(asp)
	A_q = set(gold_aspects)
	if not A_q:
		return 0.0

	denom = sum(aspect_id_to_weight.get(a, 1.0) for a in A_q)
	if denom <= 0:
		return 0.0

	covered: set = set()
	for doc_id in ranked_doc_ids[:k]:
		asp = doc_id_to_aspect_id.get(str(doc_id))
		if asp in A_q:
			covered.add(asp)
			# Early stop if all aspects are covered
			if len(covered) == len(A_q):
				break

	numer = sum(aspect_id_to_weight.get(a, 1.0) for a in covered)
	return numer / denom


def evaluate_file(
	score_file: Path,
	examples,
	doc_id_to_aspect_id: Dict[str, Optional[str]],
	aspect_id_to_weight: Dict[str, float],
	k: int,
	long_context: bool = False,
) -> Dict:
	"""
	Compute Weighted Aspect Recall@k per query for a given score.json file and summarize.
	"""
	with open(score_file) as f:
		scores = json.load(f)

	field = 'gold_ids_long' if long_context else 'gold_ids'
	values: List[float] = []
	n = 0

	for ex in examples:
		qid = str(ex['id']) if isinstance(ex['id'], (int,)) else str(ex['id'])
		if qid not in scores:
			continue
		ranked_list = parse_score_entry(scores[qid])
		gold_list = [str(x) for x in ex.get(field, [])]
		if not gold_list:
			continue

		val = compute_weighted_aspect_recall_at_k(
			ranked_list,
			gold_list,
			doc_id_to_aspect_id,
			aspect_id_to_weight,
			k=k,
		)
		values.append(val)
		n += 1

	avg_val = sum(values) / len(values) if values else 0.0
	return {
		'file': str(score_file),
		'num_evaluated_queries': n,
		'k': k,
		'avg_weighted_aspect_recall': avg_val,
	}


def find_score_files(task: str, output_dir: str, explicit_score_file: str = None, process_all: bool = False) -> List[Path]:
	if explicit_score_file:
		return [Path(explicit_score_file)]
	outputs_dir = Path(output_dir)
	if task == 'all':
		# Scan all score.json across all tasks
		return sorted([p for p in outputs_dir.glob('*/score.json')])
	if process_all:
		return [p for p in outputs_dir.glob('*/score.json') if p.parent.name.startswith(f"{task}_") or p.parent.name == task]
	# default: first matching file for task
	files = list(outputs_dir.glob(f"{task}_*/score.json"))
	if not files:
		# also allow directory exactly named task (rare)
		files = list(outputs_dir.glob(f"{task}/score.json"))
	return files[:1]


KNOWN_TASKS = [
	'biology','earth_science','economics','pony','psychology','robotics',
	'stackoverflow','sustainable_living','aops','leetcode','theoremqa_theorems',
	'theoremqa_questions'
]


def infer_task_from_dirname(dirname: str) -> Optional[str]:
	"""Infer task from run directory name using known task prefixes.

	Handles tasks containing underscores by matching the longest known task
	which is a prefix of the directory name followed by '_' (or exact match).
	"""
	for t in sorted(KNOWN_TASKS, key=len, reverse=True):
		if dirname == t or dirname.startswith(t + '_'):
			return t
	return None


def infer_retriever_from_dirname(dirname: str) -> str:
	"""Infer retriever name from a run directory name.

	Expected pattern examples:
	- biology_bm25_long_False -> retriever "bm25"
	- earth_science_bge-reasoner_long_True -> retriever "bge-reasoner"
	- psychology_grit -> retriever "grit"

	When task prefix is unknown, returns the dirname unchanged.
	"""
	task = infer_task_from_dirname(dirname)
	if task is None:
		return dirname
	# Remove the task prefix and optional underscore
	rest = dirname[len(task):]
	if rest.startswith('_'):
		rest = rest[1:]
	# Strip trailing markers like _long_True / _long_False
	if '_long_' in rest:
		rest = rest.split('_long_')[0]
	return rest if rest else dirname


def main():
	parser = argparse.ArgumentParser(description='Evaluate Weighted Aspect Recall@K over score.json files')
	parser.add_argument('--task', type=str, default='biology',
				choices=KNOWN_TASKS + ['all'])
	parser.add_argument('--k', type=int, default=25)
	parser.add_argument('--score_file', type=str, default=None)
	# Paths are relative to the current working directory (project root recommended)
	parser.add_argument('--output_dir', type=str, default='outputs')
	parser.add_argument('--cache_dir', type=str, default='cache')
	parser.add_argument('--long_context', action='store_true', help='use gold_ids_long to derive aspects-of-interest')
	parser.add_argument('--all', action='store_true', help='process all score.json files for the task (ignored when --task all)')
	parser.add_argument('--save_json', type=str, default=None, help='path to write all results as a single JSON file')
	parser.add_argument('--save_excel', type=str, default=None, help='path to write retrievers x tasks matrix to an Excel .xlsx file')
	args = parser.parse_args()

	files = find_score_files(args.task, args.output_dir, args.score_file, args.all)
	if not files:
		print(f"No score.json files found for task '{args.task}'.")
		return

	results = []
	if args.task == 'all':
		# Build per-task datasets/maps lazily and cache
		cache_per_task: Dict[str, Tuple] = {}
		for sf in files:
			dirname = sf.parent.name
			task = infer_task_from_dirname(dirname)
			if task is None:
				# Skip unknown task folders
				continue
			if task not in cache_per_task:
				examples = load_dataset('ya-ir/BRIGHT-PRO-WITH-ASPECT', 'examples', cache_dir=args.cache_dir)[task]
				doc_id_to_aspect_id, aspect_id_to_weight = build_aspect_maps(task, args.cache_dir)
				cache_per_task[task] = (examples, doc_id_to_aspect_id, aspect_id_to_weight)
			examples, doc_id_to_aspect_id, aspect_id_to_weight = cache_per_task[task]
			res = evaluate_file(sf, examples, doc_id_to_aspect_id, aspect_id_to_weight, k=args.k, long_context=args.long_context)
			# annotate result with inferred task
			res['task'] = task
			results.append(res)
	else:
		# Single task mode
		examples = load_dataset('ya-ir/BRIGHT-PRO-WITH-ASPECT', 'examples', cache_dir=args.cache_dir)[args.task]
		doc_id_to_aspect_id, aspect_id_to_weight = build_aspect_maps(args.task, args.cache_dir)
		for sf in files:
			res = evaluate_file(sf, examples, doc_id_to_aspect_id, aspect_id_to_weight, k=args.k, long_context=args.long_context)
			res['task'] = args.task
			results.append(res)

	# Print summary
	for r in results:
		print(json.dumps(r, ensure_ascii=False))

	overall = None
	if len(results) > 1:
		avg = sum(r['avg_weighted_aspect_recall'] for r in results) / len(results)
		overall = {'mode': args.task, 'files': len(results), 'avg_weighted_aspect_recall': avg, 'k': args.k}
		print(json.dumps(overall, ensure_ascii=False))

	# Per-retriever (model) averages across tasks
	retriever_to_values = {}
	for r in results:
		dirname = Path(r['file']).parent.name
		retriever = infer_retriever_from_dirname(dirname)
		retriever_to_values.setdefault(retriever, []).append(r.get('avg_weighted_aspect_recall', 0.0))

	per_retriever_average = {ret: (sum(vals) / len(vals) if vals else 0.0) for ret, vals in retriever_to_values.items()}
	if per_retriever_average:
		print(json.dumps({'k': args.k, 'per_retriever_average': per_retriever_average}, ensure_ascii=False))

	# Optional save to JSON
	if args.save_json:
		payload = {
			'mode': args.task,
			'k': args.k,
			'files': len(results),
			'results': results
		}
		if overall is not None:
			payload['aggregate'] = overall
		if per_retriever_average:
			payload['per_retriever_average'] = per_retriever_average
		with open(args.save_json, 'w') as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)

	# Optional save to Excel (retrievers as rows, tasks as columns)
	if args.save_excel:
		if pd is None:
			print("pandas is not installed; cannot write Excel. Install pandas and openpyxl to enable --save_excel.")
		else:
			# Build matrix: retriever -> task -> avg_weighted_aspect_recall
			matrix = {}
			seen_tasks = set()
			for r in results:
				dirname = Path(r['file']).parent.name
				retriever = infer_retriever_from_dirname(dirname)
				task = r.get('task') or infer_task_from_dirname(dirname) or 'unknown'
				seen_tasks.add(task)
				matrix.setdefault(retriever, {})[task] = r.get('avg_weighted_aspect_recall', None)

			# Order columns by KNOWN_TASKS where applicable
			ordered_tasks = [t for t in KNOWN_TASKS if t in seen_tasks]
			# Include any extra tasks not in KNOWN_TASKS (unlikely)
			for t in sorted(seen_tasks):
				if t not in ordered_tasks:
					ordered_tasks.append(t)

			retrievers = sorted(matrix.keys())
			data = [[matrix.get(ret, {}).get(t, None) for t in ordered_tasks] for ret in retrievers]
			df = pd.DataFrame(data=data, index=retrievers, columns=ordered_tasks)
			# Write to Excel
			with pd.ExcelWriter(args.save_excel, engine='openpyxl') as writer:
				df.to_excel(writer, sheet_name='a_recall')


if __name__ == '__main__':
	main()


