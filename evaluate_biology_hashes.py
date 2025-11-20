#!/usr/bin/env python3
"""
Script to evaluate the average number of biology-hash files in the top 25 documents
for each query across all score.json files in the outputs directory.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict


def matches_biology_hash_pattern(doc_id):
    """
    Check if a document ID matches the pattern biology-{hash}
    where hash is a hexadecimal string.
    
    Args:
        doc_id: Document identifier string
        
    Returns:
        bool: True if matches the pattern, False otherwise
    """
    # Pattern: biology- followed by a hexadecimal hash
    pattern = r'^biology-[0-9a-fA-F]+$'
    return bool(re.match(pattern, doc_id))


def process_score_file(score_file_path, top_k=25):
    """
    Process a single score.json file and return statistics.
    
    Args:
        score_file_path: Path to the score.json file
        top_k: Number of top documents to consider (default: 25)
        
    Returns:
        dict: Statistics including counts per query and averages
    """
    with open(score_file_path, 'r') as f:
        scores = json.load(f)
    
    query_counts = {}
    
    for query_id, doc_scores in scores.items():
        # Get top k documents (already sorted by score in descending order)
        top_docs = list(doc_scores.keys())[:top_k]
        
        # Count how many match the biology-hash pattern
        count = sum(1 for doc_id in top_docs if matches_biology_hash_pattern(doc_id))
        query_counts[query_id] = count
    
    # Calculate average
    total_queries = len(query_counts)
    total_count = sum(query_counts.values())
    average = total_count / total_queries if total_queries > 0 else 0
    
    return {
        'query_counts': query_counts,
        'total_queries': total_queries,
        'total_count': total_count,
        'average': average
    }


def main():
    """Main function to process all score.json files in outputs directory."""
    outputs_dir = Path('/gpfs/radev/home/jw3278/project/BRIGHT/outputs')
    
    # Find all score.json files
    score_files = list(outputs_dir.glob('*/score.json'))
    
    if not score_files:
        print("No score.json files found in outputs directory")
        return
    
    print(f"Found {len(score_files)} score.json file(s)\n")
    print("=" * 80)
    
    all_results = {}
    
    for score_file in sorted(score_files):
        experiment_name = score_file.parent.name
        print(f"\nProcessing: {experiment_name}")
        print("-" * 80)
        
        try:
            results = process_score_file(score_file)
            all_results[experiment_name] = results
            
            print(f"Total queries: {results['total_queries']}")
            print(f"Total biology-hash files in top 25: {results['total_count']}")
            print(f"Average per query: {results['average']:.4f}")
            
            # Show some example queries with their counts
            print("\nSample query counts (first 10):")
            for i, (query_id, count) in enumerate(list(results['query_counts'].items())[:10]):
                print(f"  Query {query_id}: {count} biology-hash files")
            
            if results['total_queries'] > 10:
                print(f"  ... and {results['total_queries'] - 10} more queries")
                
        except Exception as e:
            print(f"Error processing {score_file}: {e}")
            continue
    
    # Summary across all experiments
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL EXPERIMENTS")
        print("=" * 80)
        
        for exp_name, results in all_results.items():
            print(f"{exp_name:50s} | Avg: {results['average']:6.4f} | Queries: {results['total_queries']:4d}")
        
        # Overall average
        overall_total = sum(r['total_count'] for r in all_results.values())
        overall_queries = sum(r['total_queries'] for r in all_results.values())
        overall_avg = overall_total / overall_queries if overall_queries > 0 else 0
        
        print("-" * 80)
        print(f"{'OVERALL AVERAGE':50s} | Avg: {overall_avg:6.4f} | Queries: {overall_queries:4d}")
    
    print("\n" + "=" * 80)
    print("Done!")


if __name__ == '__main__':
    main()

