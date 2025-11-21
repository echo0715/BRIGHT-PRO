#!/usr/bin/env python3
"""
Script to calculate the average round of retrieval for each model and task.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import pandas as pd


def collect_retrieval_rounds(runs_dir: str) -> tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]]]:
    """
    Collect retrieved_round_count and retrieved_documents_id lengths from all JSON files in the runs directory.
    
    Args:
        runs_dir: Path to the runs directory
        
    Returns:
        Tuple of two dictionaries with structure: ({model: {task: [round_counts]}}, {model: {task: [doc_id_lengths]}})
    """
    runs_path = Path(runs_dir)
    data = defaultdict(lambda: defaultdict(list))
    doc_lengths_data = defaultdict(lambda: defaultdict(list))
    
    # Check if runs directory exists
    if not runs_path.exists():
        print(f"Error: Runs directory not found at {runs_dir}")
        return {}
    
    # Traverse the directory structure
    # Expected structure: runs/gpt-5-mini/{model}/{task}/run_*.json
    gpt_mini_dir = runs_path / "gpt-5-mini"
    
    if not gpt_mini_dir.exists():
        print(f"Error: gpt-5-mini directory not found at {gpt_mini_dir}")
        return {}
    
    # Iterate through each model directory
    for model_dir in gpt_mini_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        # Skip any backup or old directories
        if model_name.endswith('_old'):
            continue
        
        # Iterate through each task directory
        for task_dir in model_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            task_name = task_dir.name
            
            # Skip any backup or old directories
            if task_name.endswith('_old'):
                continue
            
            # Process all JSON files in the task directory
            json_files = list(task_dir.glob("run_*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                    
                    has_round_count = False
                    has_doc_ids = False
                    
                    # Extract retrieved_round_count
                    if 'retrieved_round_count' in json_data:
                        round_count = json_data['retrieved_round_count']
                        data[model_name][task_name].append(round_count)
                        has_round_count = True
                    
                    # Extract retrieved_documents_id length
                    if 'retrieved_documents_id' in json_data:
                        doc_ids = json_data['retrieved_documents_id']
                        if isinstance(doc_ids, list):
                            doc_lengths_data[model_name][task_name].append(len(doc_ids))
                            has_doc_ids = True
                        else:
                            print(f"Warning: 'retrieved_documents_id' is not a list in {json_file}")
                    
                    # Warn if both fields are missing
                    if not has_round_count and not has_doc_ids:
                        print(f"Warning: Neither 'retrieved_round_count' nor 'retrieved_documents_id' found in {json_file}")
                    elif not has_round_count:
                        print(f"Warning: 'retrieved_round_count' not found in {json_file}")
                    elif not has_doc_ids:
                        print(f"Warning: 'retrieved_documents_id' not found in {json_file}")
                        
                except json.JSONDecodeError as e:
                    print(f"Error reading {json_file}: {e}")
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
    
    return data, doc_lengths_data


def calculate_averages(data: Dict[str, Dict[str, List[int]]], 
                       doc_lengths_data: Dict[str, Dict[str, List[int]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate average retrieval rounds and document lengths for each model and task.
    
    Args:
        data: Dictionary with structure: {model: {task: [round_counts]}}
        doc_lengths_data: Dictionary with structure: {model: {task: [doc_lengths]}}
        
    Returns:
        Tuple of (rounds_detailed_df, rounds_summary_df, docs_detailed_df, docs_summary_df)
    """
    results = []
    model_summaries = []
    doc_results = []
    doc_model_summaries = []
    
    for model_name, tasks in sorted(data.items()):
        model_all_rounds = []
        
        for task_name, round_counts in sorted(tasks.items()):
            if round_counts:
                avg_rounds = sum(round_counts) / len(round_counts)
                results.append({
                    'Model': model_name,
                    'Task': task_name,
                    'Avg_Retrieval_Rounds': avg_rounds,
                    'Count': len(round_counts),
                    'Min': min(round_counts),
                    'Max': max(round_counts)
                })
                model_all_rounds.extend(round_counts)
        
        # Calculate overall average for this model
        if model_all_rounds:
            model_summaries.append({
                'Model': model_name,
                'Avg_Retrieval_Rounds': sum(model_all_rounds) / len(model_all_rounds),
                'Total_Count': len(model_all_rounds),
                'Min': min(model_all_rounds),
                'Max': max(model_all_rounds),
                'Num_Tasks': len(tasks)
            })
    
    # Calculate averages for document lengths
    for model_name, tasks in sorted(doc_lengths_data.items()):
        model_all_doc_lengths = []
        
        for task_name, doc_lengths in sorted(tasks.items()):
            if doc_lengths:
                avg_doc_length = sum(doc_lengths) / len(doc_lengths)
                doc_results.append({
                    'Model': model_name,
                    'Task': task_name,
                    'Avg_Doc_Count': avg_doc_length,
                    'Count': len(doc_lengths),
                    'Min': min(doc_lengths),
                    'Max': max(doc_lengths)
                })
                model_all_doc_lengths.extend(doc_lengths)
        
        # Calculate overall average for this model
        if model_all_doc_lengths:
            doc_model_summaries.append({
                'Model': model_name,
                'Avg_Doc_Count': sum(model_all_doc_lengths) / len(model_all_doc_lengths),
                'Total_Count': len(model_all_doc_lengths),
                'Min': min(model_all_doc_lengths),
                'Max': max(model_all_doc_lengths),
                'Num_Tasks': len(tasks)
            })
    
    detailed_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(model_summaries)
    doc_detailed_df = pd.DataFrame(doc_results)
    doc_summary_df = pd.DataFrame(doc_model_summaries)
    return detailed_df, summary_df, doc_detailed_df, doc_summary_df


def print_results(detailed_df: pd.DataFrame, summary_df: pd.DataFrame, 
                  doc_detailed_df: pd.DataFrame, doc_summary_df: pd.DataFrame):
    """
    Print the results in a readable format.
    """
    if detailed_df.empty:
        print("No data found.")
        return
    
    print("\n" + "="*80)
    print("AVERAGE RETRIEVAL ROUNDS BY MODEL AND TASK")
    print("="*80 + "\n")
    
    # Group by model
    for model in sorted(detailed_df['Model'].unique()):
        model_df = detailed_df[detailed_df['Model'] == model]
        print(f"\n{model.upper()}")
        print("-" * 80)
        print(f"{'Task':<25} {'Avg Rounds':>12} {'Count':>8} {'Min':>6} {'Max':>6}")
        print("-" * 80)
        
        for _, row in model_df.iterrows():
            print(f"{row['Task']:<25} {row['Avg_Retrieval_Rounds']:>12.2f} {row['Count']:>8} {row['Min']:>6} {row['Max']:>6}")
        
        # Overall average for this model from summary
        model_summary = summary_df[summary_df['Model'] == model].iloc[0]
        print("-" * 80)
        print(f"{'MODEL AVERAGE':<25} {model_summary['Avg_Retrieval_Rounds']:>12.2f} {model_summary['Total_Count']:>8} {int(model_summary['Min']):>6} {int(model_summary['Max']):>6}")
    
    # Model summary table
    print("\n" + "="*80)
    print("MODEL SUMMARY (AVERAGE ACROSS ALL TASKS)")
    print("="*80)
    print(f"{'Model':<15} {'Avg Rounds':>12} {'Total Runs':>12} {'Num Tasks':>12} {'Min':>6} {'Max':>6}")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<15} {row['Avg_Retrieval_Rounds']:>12.2f} {row['Total_Count']:>12} {row['Num_Tasks']:>12} {int(row['Min']):>6} {int(row['Max']):>6}")
    
    # Overall statistics across all models
    print("\n" + "="*80)
    print("OVERALL STATISTICS (ALL MODELS)")
    print("="*80)
    overall_avg = summary_df['Avg_Retrieval_Rounds'].mean()
    total_count = summary_df['Total_Count'].sum()
    print(f"Total runs: {total_count}")
    print(f"Average retrieval rounds (across all model averages): {overall_avg:.2f}")
    print("="*80 + "\n")
    
    # Print document length statistics
    if not doc_detailed_df.empty:
        print("\n" + "="*80)
        print("AVERAGE RETRIEVED DOCUMENTS COUNT BY MODEL AND TASK")
        print("="*80 + "\n")
        
        # Group by model
        for model in sorted(doc_detailed_df['Model'].unique()):
            model_df = doc_detailed_df[doc_detailed_df['Model'] == model]
            print(f"\n{model.upper()}")
            print("-" * 80)
            print(f"{'Task':<25} {'Avg Docs':>12} {'Count':>8} {'Min':>6} {'Max':>6}")
            print("-" * 80)
            
            for _, row in model_df.iterrows():
                print(f"{row['Task']:<25} {row['Avg_Doc_Count']:>12.2f} {row['Count']:>8} {row['Min']:>6} {row['Max']:>6}")
            
            # Overall average for this model from summary
            if not doc_summary_df.empty:
                model_summary = doc_summary_df[doc_summary_df['Model'] == model].iloc[0]
                print("-" * 80)
                print(f"{'MODEL AVERAGE':<25} {model_summary['Avg_Doc_Count']:>12.2f} {model_summary['Total_Count']:>8} {int(model_summary['Min']):>6} {int(model_summary['Max']):>6}")
        
        # Model summary table
        if not doc_summary_df.empty:
            print("\n" + "="*80)
            print("MODEL SUMMARY - DOCUMENT COUNTS (AVERAGE ACROSS ALL TASKS)")
            print("="*80)
            print(f"{'Model':<15} {'Avg Docs':>12} {'Total Runs':>12} {'Num Tasks':>12} {'Min':>6} {'Max':>6}")
            print("-" * 80)
            for _, row in doc_summary_df.iterrows():
                print(f"{row['Model']:<15} {row['Avg_Doc_Count']:>12.2f} {row['Total_Count']:>12} {row['Num_Tasks']:>12} {int(row['Min']):>6} {int(row['Max']):>6}")
            
            # Overall statistics across all models
            print("\n" + "="*80)
            print("OVERALL STATISTICS - DOCUMENT COUNTS (ALL MODELS)")
            print("="*80)
            overall_avg_docs = doc_summary_df['Avg_Doc_Count'].mean()
            total_count_docs = doc_summary_df['Total_Count'].sum()
            print(f"Total runs: {total_count_docs}")
            print(f"Average document count (across all model averages): {overall_avg_docs:.2f}")
            print("="*80 + "\n")


def save_results(detailed_df: pd.DataFrame, summary_df: pd.DataFrame, 
                 doc_detailed_df: pd.DataFrame, doc_summary_df: pd.DataFrame, 
                 output_dir: Path):
    """
    Save results to CSV files.
    """
    if not detailed_df.empty:
        detailed_file = output_dir / "avg_retrieval_rounds_detailed.csv"
        detailed_df.to_csv(detailed_file, index=False)
        print(f"Detailed results saved to: {detailed_file}")
    
    if not summary_df.empty:
        summary_file = output_dir / "avg_retrieval_rounds_by_model.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Model summary saved to: {summary_file}")
    
    if not doc_detailed_df.empty:
        doc_detailed_file = output_dir / "avg_doc_count_detailed.csv"
        doc_detailed_df.to_csv(doc_detailed_file, index=False)
        print(f"Document count detailed results saved to: {doc_detailed_file}")
    
    if not doc_summary_df.empty:
        doc_summary_file = output_dir / "avg_doc_count_by_model.csv"
        doc_summary_df.to_csv(doc_summary_file, index=False)
        print(f"Document count model summary saved to: {doc_summary_file}")


def main():
    # Get the runs directory path
    script_dir = Path(__file__).parent
    runs_dir = script_dir.parent / "runs"
    
    print(f"Scanning directory: {runs_dir}")
    print("Please wait...")
    
    # Collect data
    data, doc_lengths_data = collect_retrieval_rounds(str(runs_dir))
    
    if not data:
        print("No data collected. Please check the directory structure.")
        return
    
    # Calculate averages
    detailed_df, summary_df, doc_detailed_df, doc_summary_df = calculate_averages(data, doc_lengths_data)
    
    # Print results
    print_results(detailed_df, summary_df, doc_detailed_df, doc_summary_df)
    
    # Save to CSV (save to parent directory to keep outputs at root level)
    save_results(detailed_df, summary_df, doc_detailed_df, doc_summary_df, script_dir.parent)


if __name__ == "__main__":
    main()

