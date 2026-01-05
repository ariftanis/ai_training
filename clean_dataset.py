import json
import sys
from typing import Tuple

def clean_dataset(input_file: str, cleaned_file: str, rejected_file: str) -> Tuple[int, int]:
    """
    Cleans the dataset by separating valid and invalid lines.
    
    Args:
        input_file: Path to the input dataset file
        cleaned_file: Path to save the cleaned dataset
        rejected_file: Path to save the rejected lines
    
    Returns:
        Tuple of (valid_lines_count, rejected_lines_count)
    """
    print(f"Cleaning dataset: {input_file}")
    print(f"Valid entries will be saved to: {cleaned_file}")
    print(f"Rejected entries will be saved to: {rejected_file}")
    
    valid_count = 0
    rejected_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(cleaned_file, 'w', encoding='utf-8') as clean_outfile, \
         open(rejected_file, 'w', encoding='utf-8') as reject_outfile:
        
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the JSON
                data = json.loads(line)
                
                # Check if required fields exist and are strings
                required_fields = ['instruction', 'input', 'output']
                is_valid = True
                
                for field in required_fields:
                    if field not in data:
                        print(f"Line {line_num}: Missing required field '{field}'")
                        is_valid = False
                        break
                    elif not isinstance(data[field], str):
                        print(f"Line {line_num}: Field '{field}' is not a string (type: {type(data[field])})")
                        is_valid = False
                        break
                
                if is_valid:
                    # Write to cleaned dataset
                    clean_outfile.write(line + '\n')
                    valid_count += 1
                    if valid_count % 5000 == 0:
                        print(f"Processed {valid_count} valid entries...")
                else:
                    # Write to rejected dataset
                    reject_outfile.write(f"Line {line_num}: {line}\n")
                    rejected_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")
                reject_outfile.write(f"Line {line_num}: {line}\n")
                rejected_count += 1
            except Exception as e:
                print(f"Line {line_num}: Unexpected error - {e}")
                reject_outfile.write(f"Line {line_num}: {line}\n")
                rejected_count += 1
    
    return valid_count, rejected_count

def main():
    input_file = 'dataset.jsonl'
    cleaned_file = 'cleaned_dataset.jsonl'
    rejected_file = 'rejected_dataset.jsonl'
    
    print("Starting dataset cleaning process...")
    valid_count, rejected_count = clean_dataset(input_file, cleaned_file, rejected_file)
    
    print(f"\nDataset cleaning completed!")
    print(f"Valid entries: {valid_count}")
    print(f"Rejected entries: {rejected_count}")
    print(f"Total processed: {valid_count + rejected_count}")
    
    if rejected_count == 0:
        print("\n✅ No issues found. The dataset is clean and ready for training.")
    else:
        print(f"\n⚠️  Issues found. Cleaned dataset saved to '{cleaned_file}'")
        print(f"Rejected entries saved to '{rejected_file}' for review and correction.")
    
    return rejected_count == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)