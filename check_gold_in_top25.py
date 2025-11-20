#!/usr/bin/env python3
"""
Script to evaluate how many gold documents appear in the top 25 retrieved documents
for each query across score.json files.
"""

import json
import os
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def check_gold_in_top_k(score_file_path, examples, top_k=25, long_context=False):
    """
    Check how many gold documents appear in the top-k retrieved documents for each query.
    
    Args:
        score_file_path: Path to the score.json file
        examples: Dataset containing queries with gold_ids
        top_k: Number of top documents to consider (default: 25)
        long_context: Whether to use gold_ids_long (default: False)
        
    Returns:
        dict: Statistics including counts per query and averages
    """
    # Load scores
    with open(score_file_path, 'r') as f:
        scores = json.load(f)
    
    # Build gold_ids mapping from examples
    gold_ids_map = {}
    key = 'gold_ids_long' if long_context else 'gold_ids'
    
    for example in examples:
        query_id = str(example['id'])
        gold_ids_map[query_id] = set(example[key])
    
    # Check each query
    query_gold_counts = {}
    
    for query_id, doc_scores in tqdm(scores.items(), desc="Processing queries"):
        # Get top k documents (scores are already sorted by score in descending order)
        top_docs = list(doc_scores.keys())[:top_k]
        
        # Count how many of the top docs are gold documents
        if query_id in gold_ids_map:
            gold_docs = gold_ids_map[query_id]
            count = sum(1 for doc_id in top_docs if doc_id in gold_docs)
            query_gold_counts[query_id] = count
        else:
            print(f"Warning: Query {query_id} not found in examples dataset")
            query_gold_counts[query_id] = 0
    
    # Calculate statistics
    total_queries = len(query_gold_counts)
    total_gold_found = sum(query_gold_counts.values())
    average_gold_per_query = total_gold_found / total_queries if total_queries > 0 else 0
    
    # Distribution statistics
    counts_distribution = {}
    for count in query_gold_counts.values():
        counts_distribution[count] = counts_distribution.get(count, 0) + 1
    
    return {
        'query_gold_counts': query_gold_counts,
        'total_queries': total_queries,
        'total_gold_found': total_gold_found,
        'average_gold_per_query': average_gold_per_query,
        'counts_distribution': counts_distribution
    }


def main():
    """Main function to process score.json files."""
    parser = argparse.ArgumentParser(description='Check how many gold documents are in top-K results')
    parser.add_argument('--task', type=str, default='biology',
                        choices=['biology','earth_science','economics','pony','psychology','robotics',
                                 'stackoverflow','sustainable_living','aops','leetcode','theoremqa_theorems',
                                 'theoremqa_questions'],
                        help='Task name (default: biology)')
    parser.add_argument('--score_file', type=str, default=None,
                        help='Path to specific score.json file (if not provided, will search in outputs)')
    parser.add_argument('--output_dir', type=str, default='/gpfs/radev/home/jw3278/project/BRIGHT/outputs',
                        help='Directory containing score.json files')
    parser.add_argument('--cache_dir', type=str, default='/gpfs/radev/home/jw3278/project/BRIGHT/cache',
                        help='Cache directory for datasets')
    parser.add_argument('--top_k', type=int, default=25,
                        help='Number of top documents to check (default: 25)')
    parser.add_argument('--long_context', action='store_true',
                        help='Use gold_ids_long instead of gold_ids')
    parser.add_argument('--all', action='store_true',
                        help='Process all score.json files for the task in output_dir')
    args = parser.parse_args()
    
    # Load examples dataset
    print(f"Loading examples dataset for task: {args.task}")
    examples = load_dataset('ya-ir/BRIGHT-PRO-WITH-ASPECT', 'examples', cache_dir=args.cache_dir)[args.task]
    print(f"Loaded {len(examples)} examples\n")
    
    # Determine which score files to process
    if args.score_file:
        score_files = [Path(args.score_file)]
    elif args.all:
        # Find all score.json files matching the task
        outputs_dir = Path(args.output_dir)
        score_files = [f for f in outputs_dir.glob('*/score.json') if args.task in f.parent.name]
        if not score_files:
            print(f"No score.json files found for task '{args.task}' in {outputs_dir}")
            return
        print(f"Found {len(score_files)} score.json file(s) for task '{args.task}'\n")
    else:
        # Default: use the specific output directory pattern
        score_file_path = Path(args.output_dir) / f"{args.task}_*" / "score.json"
        # Try to find a matching file
        score_files = list(Path(args.output_dir).glob(f"{args.task}_*/score.json"))
        if not score_files:
            print(f"No score.json files found for task '{args.task}'")
            print(f"Please specify --score_file or use --all to process all files")
            return
        if len(score_files) == 1:
            print(f"Found 1 score.json file for task '{args.task}'\n")
        else:
            print(f"Found {len(score_files)} score.json files. Use --all to process all or specify --score_file for a specific one\n")
            score_files = [score_files[0]]
            print(f"Processing only: {score_files[0]}\n")
    
    # Process each score file
    all_results = {}
    
    for score_file in sorted(score_files):
        experiment_name = score_file.parent.name
        print("=" * 80)
        print(f"Processing: {experiment_name}")
        print(f"Score file: {score_file}")
        print("=" * 80)
        
        try:
            results = check_gold_in_top_k(
                score_file_path=score_file,
                examples=examples,
                top_k=args.top_k,
                long_context=args.long_context
            )
            all_results[experiment_name] = results
            
            print(f"\nResults:")
            print(f"  Total queries: {results['total_queries']}")
            print(f"  Total gold documents found in top-{args.top_k}: {results['total_gold_found']}")
            print(f"  Average gold documents per query: {results['average_gold_per_query']:.4f}")
            
            # Show distribution
            print(f"\n  Distribution of gold documents found:")
            for count in sorted(results['counts_distribution'].keys()):
                num_queries = results['counts_distribution'][count]
                percentage = (num_queries / results['total_queries']) * 100
                print(f"    {count} gold docs: {num_queries} queries ({percentage:.1f}%)")
            
            # Show some example queries with their counts
            print(f"\n  Sample query counts (first 10):")
            for i, (query_id, count) in enumerate(list(results['query_gold_counts'].items())[:10]):
                print(f"    Query {query_id}: {count} gold documents in top-{args.top_k}")
            
            if results['total_queries'] > 10:
                print(f"    ... and {results['total_queries'] - 10} more queries")
            
            print()
                
        except Exception as e:
            print(f"Error processing {score_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary across all experiments
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL EXPERIMENTS")
        print("=" * 80)
        
        for exp_name, results in all_results.items():
            print(f"{exp_name:60s} | Avg: {results['average_gold_per_query']:6.4f} | Queries: {results['total_queries']:4d}")
        
        # Overall average
        overall_total = sum(r['total_gold_found'] for r in all_results.values())
        overall_queries = sum(r['total_queries'] for r in all_results.values())
        overall_avg = overall_total / overall_queries if overall_queries > 0 else 0
        
        print("-" * 80)
        print(f"{'OVERALL AVERAGE':60s} | Avg: {overall_avg:6.4f} | Queries: {overall_queries:4d}")
    
    print("\n" + "=" * 80)
    print("Done!")


if __name__ == '__main__':
    main()


