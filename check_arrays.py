import json
import sys

def check_array_fields(file_path):
    """
    Checks for any fields that contain arrays instead of strings in the dataset
    """
    print(f"Checking for array fields in: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Check if any of the required fields are arrays instead of strings
                for field in ['instruction', 'input', 'output']:
                    if field in data:
                        if isinstance(data[field], list):
                            print(f"Line {line_num}: Field '{field}' is an array instead of a string")
                            print(f"Content: {data[field]}")
                            return False
                        elif not isinstance(data[field], str):
                            print(f"Line {line_num}: Field '{field}' is not a string")
                            print(f"Type: {type(data[field])}, Content: {data[field]}")
                            return False
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")
                return False
    
    print("No array fields found. All fields are strings as expected.")
    return True

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'dataset.jsonl'
    
    success = check_array_fields(file_path)
    sys.exit(0 if success else 1)