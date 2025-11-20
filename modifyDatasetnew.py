'''
Create the new BRIGHT dataset with filtered extraction results from o4-mini for our data, and from gpt4.1 and o4-mini for the original BRIGHT datasets.
'''


from datasets import load_dataset, Dataset, DatasetDict
import json
import os
import glob
import re
import uuid
from typing import Dict, List, Tuple

def extract_file_info(filename: str) -> Tuple[str, str]:
    """
    Extract category and index from filename like 'biology-68_extractions.json'
    Returns: (category, index)
    """
    # Remove the '_extractions.json' suffix
    base_name = filename.replace('_extractions.json', '')
    
    # Split by '-' to separate category and index
    parts = base_name.split('-')
    if len(parts) >= 2:
        category = parts[0]
        index = parts[1]
        return category, index
    else:
        return None, None

def generate_document_id(category: str, question_id: str, extraction_index: int) -> str:
    """
    Generate a unique document ID for each extraction
    Format: {category}_{question_id}_extraction_{extraction_index}
    """
    return f"{category}_{question_id}_extraction_{extraction_index}"

def load_and_process_extractions(extraction_dir: str = "filtered_extraction_results_combined") -> Dict[str, List[Dict]]:
    """
    Load all extraction files and organize them by category and question_id
    Returns: Dict mapping (category, question_id) to list of extractions
    """
    print("=== DEBUG: Loading extraction files... ===")
    print(f"Looking for files in directory: {extraction_dir}")
    
    extraction_data = {}
    json_files = glob.glob(os.path.join(extraction_dir, "*.json"))
    
    print(f"Found {len(json_files)} JSON files: {json_files}")
    
    if not json_files:
        print("ERROR: No JSON files found in extraction directory!")
        return extraction_data
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        category, index = extract_file_info(filename)
        
        if category is None or index is None:
            print(f"Warning: Could not parse filename {filename}")
            continue
            
        print(f"Processing: {filename} -> category: {category}, index: {index}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        
        extractions = data
        question_id = extractions[0].get('question_id', f"{category}-{index}")
        
        print(f"  question_id: {question_id}")
        print(f"  number of extractions: {len(extractions)}")
        
        key = (category, question_id)
        if key not in extraction_data:
            extraction_data[key] = []
            
        # Add extraction index to each extraction
        for i, extraction in enumerate(extractions):
            extraction['extraction_index'] = i
            extraction['source_file'] = filename
            extraction['category'] = category
            extraction['question_id'] = question_id
            extraction_data[key].append(extraction)
    
    print(f"=== DEBUG: Loaded {len(extraction_data)} question groups with extractions ===")
    for key, extractions in extraction_data.items():
        print(f"  {key}: {len(extractions)} extractions")
    
    return extraction_data

def modify_dataset_with_extractions():
    """
    Main function to modify the BRIGHT dataset with extraction contents
    """
    print("=== DEBUG: Starting dataset modification... ===")
    
    # Load extraction data
    extraction_data = load_and_process_extractions()
    
    if not extraction_data:
        print("ERROR: No extraction data loaded! Exiting.")
        return None
    
    # Load the BRIGHT dataset
    print("=== DEBUG: Loading BRIGHT dataset... ===")
    try:
        # Load the examples and documents separately with proper configs
        examples_dataset = load_dataset('xlangai/BRIGHT', 'examples')
        documents_dataset = load_dataset('xlangai/BRIGHT', 'documents')
        print("SUCCESS: BRIGHT dataset loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load BRIGHT dataset: {e}")
        return None
    
    # Get available splits from examples dataset
    available_splits = list(examples_dataset.keys())
    print(f"Available splits: {available_splits}")
    
    # Initialize the correct structure: subsets at top level, splits nested
    modified_examples = {}
    modified_documents = {}
    
    # Process each split
    for split_name in available_splits:
        print(f"\n=== DEBUG: Processing split: {split_name} ===")
        
        # Get examples and documents for this split
        if split_name in examples_dataset:
            examples = examples_dataset[split_name]
            print(f"Found {len(examples)} examples in {split_name}")
        else:
            print(f"No examples found in {split_name}")
            continue
            
        if split_name in documents_dataset:
            documents = documents_dataset[split_name]
            print(f"Found {len(documents)} documents in {split_name}")
        else:
            print(f"No documents found in {split_name}")
            continue
        
        # Convert to list for easier manipulation
        if hasattr(examples, 'to_list'):
            examples_list = examples.to_list()
        else:
            examples_list = list(examples)
        
        if hasattr(documents, 'to_list'):
            documents_list = documents.to_list()
        else:
            documents_list = list(documents)
        
        # Create mappings for efficient lookup - use numeric IDs as keys
        examples_by_id = {str(ex['id']): ex for ex in examples_list}
        print(f"Created lookup table with {len(examples_by_id)} examples")
        
        # Show some example IDs for debugging
        sample_ids = list(examples_by_id.keys())[:5]
        print(f"Sample example IDs: {sample_ids}")
        
        # Track new documents and modified examples
        new_documents = []
        modified_examples_list = []
        
        # Process extractions for this split
        print(f"Processing extractions for category: {split_name}")
        matches_found = 0
        
        # Find extractions that match this category
        category_extractions = {}
        for (category, question_id), extractions in extraction_data.items():
            if category == split_name:
                # Extract the numeric part from question_id (e.g., "biology-98" -> "98")
                if '-' in question_id:
                    try:
                        numeric_id = question_id.split('-')[1]
                        category_extractions[numeric_id] = extractions
                    except:
                        print(f"Could not parse numeric ID from {question_id}")
        
        print(f"Found {len(category_extractions)} extraction groups for category {split_name}")

        if len(category_extractions) == 0:
            print(f"No extractions found for category {split_name}")
            modified_documents[split_name] = Dataset.from_list(documents_list)
            modified_examples[split_name] = Dataset.from_list(examples_list)
            continue
            

        
        # Process each extraction group
        for numeric_id, extractions in category_extractions.items():
            print(f"  Checking numeric_id: {numeric_id}")
            
            # Check if this numeric_id exists in the examples
            if numeric_id in examples_by_id:
                matches_found += 1
                example = examples_by_id[numeric_id]
                print(f"  ✓ MATCH FOUND! Processing extractions for question {numeric_id} ({len(extractions)} extractions)")
                
                # Show original gold_ids
                original_gold_ids = example.get('gold_ids', [])
                print(f"    Original gold_ids: {len(original_gold_ids)} items")
                
                # Clear existing gold_ids to start fresh with only new extraction IDs
                example['gold_ids'] = []


                # Add each extraction as a new document
                for extraction in extractions:
                    # Generate unique document ID

                    doc_id = extraction['gold_id']
                    print(f"    Creating document: {doc_id}")

                    response = json.loads(extraction['validation_response'])

                    # If the extraction is not relevant, skip it
                    if not response['is_relevant']:
                        continue
                    
                    # Create new document entry
                    new_document = {
                        'id': doc_id,
                        'content': extraction['extracted_content'],
                    }

                    if "aspect" in extraction:
                        new_documents.append(new_document)
                    
                    # Add document ID to the example's gold_ids
                    # if 'gold_ids' in example:
                    #     example['gold_ids'].append(doc_id)
                    # else:
                    #     example['gold_ids'] = [doc_id]
                    example['gold_ids'].append(doc_id)
                
                # Show updated gold_ids
                updated_gold_ids = example.get('gold_ids', [])
                print(f"    Updated gold_ids: {len(updated_gold_ids)} items (+{len(updated_gold_ids) - len(original_gold_ids)})")
                
                # Add modified example to the list
                modified_examples_list.append(example)
            else:
                print(f"  ✗ No match found for numeric_id: {numeric_id}")
        
        print(f"Total matches found: {matches_found} out of {len(category_extractions)} extraction groups")
        
        # Create new datasets for this split
        if new_documents:
            print(f"Adding {len(new_documents)} new documents to {split_name}")
            all_documents = documents_list + new_documents
            # all_documents = new_documents
            modified_documents[split_name] = Dataset.from_list(all_documents)
        else:
            print(f"No new documents to add for {split_name}")
            modified_documents[split_name] = documents
            
        if modified_examples_list:
            print(f"Modified {len(modified_examples_list)} examples in {split_name}")
            modified_examples[split_name] = Dataset.from_list(modified_examples_list)
        else:
            print(f"No examples modified for {split_name}")
            modified_examples[split_name] = examples
    
    # Create the final modified dataset with correct structure
    # Subsets at top level, splits nested under each subset
    print("\n=== DEBUG: Creating final dataset structure ===")
    final_dataset = DatasetDict({
        'examples': DatasetDict(modified_examples),
        'documents': DatasetDict(modified_documents)
    })
    
    # Save the modified dataset
    output_path = "bright_pro"
    print(f"\n=== DEBUG: Saving modified dataset to {output_path} ===")
    try:
        final_dataset.save_to_disk(output_path)
        print("SUCCESS: Dataset saved successfully!")
    except Exception as e:
        print(f"ERROR: Failed to save dataset: {e}")
        return None


    # Print summary statistics
    print("\n=== Summary ===")
    print("Examples subset:")
    for split_name, split_data in final_dataset['examples'].items():
        num_examples = len(split_data)
        print(f"  {split_name}: {num_examples} examples")
        
        # Count average gold_ids per example
        if num_examples > 0:
            total_gold_ids = sum(len(ex.get('gold_ids', [])) for ex in split_data)
            avg_gold_ids = total_gold_ids / num_examples
            print(f"    Average gold_ids per example: {avg_gold_ids:.2f}")
    
    print("\nDocuments subset:")
    for split_name, split_data in final_dataset['documents'].items():
        num_documents = len(split_data)
        print(f"  {split_name}: {num_documents} documents")
    
    return final_dataset


if __name__ == '__main__':
    # Run the main modification function
    print("=== DEBUG: Starting main execution ===")
    modified_dataset = modify_dataset_with_extractions()
    if modified_dataset:
        print("=== DEBUG: Dataset modification completed successfully! ===")
    else:
        print("=== DEBUG: Dataset modification failed! ===")
