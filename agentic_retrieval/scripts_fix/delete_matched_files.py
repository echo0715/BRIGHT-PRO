#!/usr/bin/env python3
"""
Script to delete files listed in the matched_files array of a JSON file.
"""

import json
import os
from pathlib import Path

def delete_matched_files(json_path):
    """
    Read JSON file and delete all files listed in the 'matched_files' array.
    
    Args:
        json_path: Path to the JSON file containing matched_files
    """
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    matched_files = data.get('matched_files', [])
    total_files = len(matched_files)
    deleted_count = 0
    not_found_count = 0
    error_count = 0
    
    print(f"Found {total_files} files to delete")
    print("-" * 60)
    
    for file_path in matched_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_count += 1
                print(f"✓ Deleted: {file_path}")
            else:
                not_found_count += 1
                print(f"✗ Not found: {file_path}")
        except Exception as e:
            error_count += 1
            print(f"✗ Error deleting {file_path}: {e}")
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total files: {total_files}")
    print(f"  Successfully deleted: {deleted_count}")
    print(f"  Not found: {not_found_count}")
    print(f"  Errors: {error_count}")

if __name__ == "__main__":
    json_file = "grouped_tool_calls.json"
    
    # Get the script's directory and construct the JSON path
    script_dir = Path(__file__).parent
    json_path = script_dir / json_file
    
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        exit(1)
    
    # Ask for confirmation before deleting
    response = input(f"Are you sure you want to delete {json_file}'s matched files? (yes/no): ")
    if response.lower() == 'yes':
        delete_matched_files(json_path)
    else:
        print("Operation cancelled.")





