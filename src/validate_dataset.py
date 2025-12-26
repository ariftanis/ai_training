import json
import os
from collections import Counter

def validate_dataset():
    """
    Validates the dataset for quality and correctness
    """
    dataset_file = 'dataset.jsonl'
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file '{dataset_file}' does not exist.")
        print("Please run prepare_dataset.py first.")
        return False
    
    print("Validating dataset...")
    
    # Load the dataset
    records = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Total number of records: {len(records)}")
    
    # Check field presence
    required_fields = ['instruction', 'input', 'output']
    missing_fields = []
    
    for i, record in enumerate(records):
        for field in required_fields:
            if field not in record:
                missing_fields.append(f"Record {i}: Missing '{field}' field")
    
    if missing_fields:
        print(f"Found {len(missing_fields)} records with missing fields:")
        for field in missing_fields[:10]:  # Show first 10
            print(f"  - {field}")
        return False
    else:
        print("✓ All records have required fields")
    
    # Analyze data quality
    instructions = [r['instruction'] for r in records]
    inputs = [r['input'] for r in records]
    outputs = [r['output'] for r in records]
    
    # Check for empty fields
    empty_records = []
    for i, record in enumerate(records):
        for field in ['instruction', 'input', 'output']:
            if not record[field].strip():
                empty_records.append(f"Record {i}: Empty '{field}' field")
    
    if empty_records:
        print(f"Found {len(empty_records)} records with empty fields:")
        for record in empty_records[:10]:  # Show first 10
            print(f"  - {record}")
        return False
    else:
        print("✓ No records have empty fields")
    
    # Check for duplicate records
    record_texts = [f"{r['input']}_{r['output']}" for r in records]
    unique_records = set(record_texts)
    
    if len(unique_records) != len(records):
        duplicates = len(records) - len(unique_records)
        print(f"⚠ Found {duplicates} duplicate records")
    else:
        print("✓ No duplicate records found")
    
    # Analyze content lengths
    input_lengths = [len(inp) for inp in inputs]
    output_lengths = [len(out) for out in outputs]
    
    avg_input_len = sum(input_lengths) / len(input_lengths) if input_lengths else 0
    avg_output_len = sum(output_lengths) / len(output_lengths) if output_lengths else 0
    
    print(f"Average input length: {avg_input_len:.2f} characters")
    print(f"Average output length: {avg_output_len:.2f} characters")
    
    # Check for extremely short/long records
    short_inputs = [i for i, l in enumerate(input_lengths) if l < 10]
    long_inputs = [i for i, l in enumerate(input_lengths) if l > 500]
    short_outputs = [i for i, l in enumerate(output_lengths) if l < 10]
    long_outputs = [i for i, l in enumerate(output_lengths) if l > 2000]
    
    if short_inputs:
        print(f"⚠ Found {len(short_inputs)} very short inputs (less than 10 chars)")
    if long_inputs:
        print(f"⚠ Found {len(long_inputs)} very long inputs (more than 500 chars)")
    if short_outputs:
        print(f"⚠ Found {len(short_outputs)} very short outputs (less than 10 chars)")
    if long_outputs:
        print(f"⚠ Found {len(long_outputs)} very long outputs (more than 2000 chars)")
    
    # Check instruction diversity
    instruction_counts = Counter(instructions)
    print(f"Number of unique instructions: {len(instruction_counts)}")
    print("Most common instructions:")
    for instruction, count in instruction_counts.most_common(5):
        print(f"  - '{instruction}' ({count} times)")
    
    # Sample some records for manual review
    print("\nSample records for manual review:")
    for i in range(min(3, len(records))):
        print(f"\nRecord {i+1}:")
        print(f"  Instruction: {records[i]['instruction']}")
        print(f"  Input: {records[i]['input'][:100]}...")
        print(f"  Output: {records[i]['output'][:150]}...")
    
    print(f"\n✓ Dataset validation completed successfully")
    print(f"✓ Dataset is ready for training with {len(records)} QA pairs")
    
    return True

def check_data_quality():
    """
    Additional checks for data quality
    """
    dataset_file = 'dataset.jsonl'
    
    if not os.path.exists(dataset_file):
        return False

    records = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # Check for hallucinations in outputs
    potential_hallucinations = 0
    for record in records:
        output = record['output'].lower()
        # Look for phrases indicating uncertainty or hallucination
        uncertainty_phrases = [
            'i don\'t know', 'unknown', 'not specified', 'not mentioned', 
            'not provided', 'cannot determine', 'not clear'
        ]
        for phrase in uncertainty_phrases:
            if phrase in output:
                potential_hallucinations += 1
                break
    
    print(f"Records with potential uncertainty/hallucination indicators: {potential_hallucinations}")
    
    return True

if __name__ == '__main__':
    success = validate_dataset()
    if success:
        check_data_quality()
        print("\n🎉 Dataset validation completed! The dataset is ready for training.")
    else:
        print("\n❌ Dataset validation failed. Please fix the issues and regenerate the dataset.")