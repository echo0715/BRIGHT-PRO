import argparse
import glob
import json
import math
import os
from typing import Dict, List, Set, Tuple
from datasets import load_dataset


def load_gold_from_dataset(task: str, cache_dir: str | None = None) -> Dict[str, Set[str]]:
    """Load ground-truth relevance from BRIGHT-PRO examples split.

    For each example, `id` is the query id and `gold_ids` is a list of relevant doc ids.
    """
    ds = load_dataset('ya-ir/BRIGHT-PRO', 'examples', cache_dir=cache_dir)[task]
    gold_map: Dict[str, Set[str]] = {}
    for ex in ds:
        qid = str(ex.get('id'))
        gold_ids = ex.get('gold_ids') or []
        gold_map[qid] = {str(d) for d in gold_ids}
    return gold_map


def dcg_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(ranked_ids))):
        rel_i = 1.0 if ranked_ids[i] in relevant else 0.0
        # log2(i+2) because rank i is 0-based here, denominator uses log2(position+1)
        denom = math.log2(i + 2)
        dcg += (2.0 ** rel_i - 1.0) / denom
    return dcg


def idcg_at_k(num_relevant: int, k: int) -> float:
    ideal_hits = min(num_relevant, k)
    idcg = 0.0
    for i in range(ideal_hits):
        idcg += (2.0 ** 1.0 - 1.0) / math.log2(i + 2)
    return idcg


def ndcg_at_k(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    dcg = dcg_at_k(ranked_ids, relevant, k)
    idcg = idcg_at_k(len(relevant), k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def auc_ndcg(ranked_ids: List[str], relevant: Set[str], kmax: int) -> float:
    if kmax <= 0:
        return 0.0
    total = 0.0
    for k in range(1, kmax + 1):
        total += ndcg_at_k(ranked_ids, relevant, k)
    return total / float(kmax)


def evaluate_runs(
    runs_dir: str,
    task: str,
    cache_dir: str | None = None,
    per_file: bool = False,
) -> Tuple[float, List[Tuple[str, float]], int, int, int]:
    qrels = load_gold_from_dataset(task=task, cache_dir=cache_dir)

    run_paths = sorted(glob.glob(os.path.join(runs_dir, "run_*.json")))

    # Determine global Kmax across all runs (maximum retrieved list length)
    kmax_global = 0
    for rp in run_paths:
        try:
            with open(rp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        ranked_ids_raw = data.get("retrieved_documents_id", []) or []
        # Count items; normalization to strings isn't needed for length
        kmax_global = max(kmax_global, len([x for x in ranked_ids_raw if x is not None]))
    print(f"Global Kmax: {kmax_global}")

    per_file_scores: List[Tuple[str, float]] = []
    num_with_rels = 0
    num_nonzero = 0

    auc_values: List[float] = []

    for rp in run_paths:
        try:
            with open(rp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        qid = str(data.get("query_id", "")).strip()
        ranked_ids_raw = data.get("retrieved_documents_id", []) or []

        # Normalize all ids to strings
        ranked_ids = [str(x).strip() for x in ranked_ids_raw if x is not None]

        # Check qrels and count files with ground truth
        relevant = qrels.get(qid, set())
        if relevant:
            num_with_rels += 1
        else:
            print(f"WARNING: No qrels found for query_id '{qid}' in file {os.path.basename(rp)}")
        
        if not ranked_ids:
            print(f"INFO: Empty retrieved_documents_id for query_id '{qid}' in file {os.path.basename(rp)}")
            score = 0.0
            per_file_scores.append((os.path.basename(rp), score))
            auc_values.append(score)
            continue

        score = auc_ndcg(ranked_ids, relevant, kmax_global)
        if score > 0:
            num_nonzero += 1

        per_file_scores.append((os.path.basename(rp), score))
        auc_values.append(score)

    macro_avg = sum(auc_values) / len(auc_values) if auc_values else 0.0
    if not per_file:
        per_file_scores = []
    return macro_avg, per_file_scores, len(run_paths), num_with_rels, num_nonzero


def main():
    parser = argparse.ArgumentParser(
        description="Compute AUC-NDCG over retrieved_documents_id using BRIGHT-PRO gold_ids and macro-average across files."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., reasonir, grit, diver) - will construct path as runs/gpt-5-mini/{model}/{task}",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name (e.g., biology, earth_science, etc.) to index into dataset split. If not provided, runs for all tasks.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="HF datasets cache dir",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print per-file AUC-NDCG scores",
    )

    args = parser.parse_args()

    # Define all tasks
    all_tasks = [
        "biology",
        "earth_science",
        "economics",
        "psychology",
        "robotics",
        "stackoverflow",
        "sustainable_living"
    ]

    # Determine which tasks to run
    tasks_to_run = [args.task] if args.task else all_tasks

    # Store results for all tasks
    all_results = []

    for task in tasks_to_run:
        # Construct runs directory path
        runs_dir = f"runs/gpt-5-mini/{args.model}/{task}"

        macro_avg, per_file_scores, total, with_rels, nonzero = evaluate_runs(
            runs_dir=runs_dir,
            task=task,
            cache_dir=args.cache_dir,
            per_file=args.per_file,
        )

        all_results.append({
            "task": task,
            "macro_avg": macro_avg,
            "total": total,
            "with_rels": with_rels,
            "nonzero": nonzero,
            "per_file_scores": per_file_scores,
        })

        print(f"\nModel:    {args.model}")
        print(f"Task:     {task}")
        print(f"Runs dir: {runs_dir}")
        print(f"Files:    {total} | With qrels: {with_rels} | Nonzero AUC: {nonzero}")
        print(f"Macro AUC-NDCG: {macro_avg:.6f}")

        if per_file_scores:
            print("\nPer-file AUC-NDCG:")
            for name, s in per_file_scores:
                print(f"{name}\t{s:.6f}")

    # If running all tasks, print summary
    if len(tasks_to_run) > 1:
        print("\n" + "="*60)
        print("SUMMARY ACROSS ALL TASKS")
        print("="*60)
        print(f"Model: {args.model}")
        for result in all_results:
            print(f"{result['task']:20s}: {result['macro_avg']:.6f} (Files: {result['total']}, With qrels: {result['with_rels']}, Nonzero: {result['nonzero']})")
        
        # Compute overall average
        valid_results = [r["macro_avg"] for r in all_results if r["total"] > 0]
        overall_avg = sum(valid_results) / len(valid_results) if valid_results else 0.0
        print(f"\n{'Overall Average':20s}: {overall_avg:.6f}")


if __name__ == "__main__":
    main()


