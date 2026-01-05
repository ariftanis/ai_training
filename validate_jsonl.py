import json
import sys

def validate_jsonl_file(file_path):
    """
    Validates a JSONL file to ensure each line is valid JSON.
    """
    print(f"Validating JSONL file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        line_number = 0
        valid_lines = 0
        invalid_lines = 0
        
        for line in file:
            line_number += 1
            line = line.strip()
            
            if not line:
                continue  # Skip empty lines
            
            try:
                # Try to parse the line as JSON
                json_obj = json.loads(line)
                
                # Check if required fields exist (for this specific dataset)
                required_fields = ['instruction', 'input', 'output']
                missing_fields = [field for field in required_fields if field not in json_obj]
                
                if missing_fields:
                    print(f"Line {line_number}: Missing required fields: {missing_fields}")
                    invalid_lines += 1
                else:
                    valid_lines += 1
                    
            except json.JSONDecodeError as e:
                print(f"Line {line_number}: Invalid JSON - {e}")
                print(f"Content: {line[:100]}...")  # Print first 100 chars of problematic line
                invalid_lines += 1
            except Exception as e:
                print(f"Line {line_number}: Unexpected error - {e}")
                invalid_lines += 1
    
    print(f"\nValidation complete:")
    print(f"Valid lines: {valid_lines}")
    print(f"Invalid lines: {invalid_lines}")
    print(f"Total processed lines: {valid_lines + invalid_lines}")
    
    if invalid_lines == 0:
        print("\n[SUCCESS] The file is valid JSONL format!")
        return True
    else:
        print(f"\n[ERROR] Found {invalid_lines} invalid lines in the JSONL file.")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'dataset.jsonl'
    
    success = validate_jsonl_file(file_path)
    sys.exit(0 if success else 1)