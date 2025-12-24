import os
import json

def create_dataset():
    """
    Reads all .md and .json files from the 'documents' directory,
    converts them into the Alpaca JSONL format, and saves the output
    to 'dataset.jsonl' in the root directory.
    """
    documents_dir = 'documents'
    output_file = 'dataset.jsonl'
    dataset = []

    if not os.path.exists(documents_dir):
        print(f"Error: The '{documents_dir}' directory does not exist.")
        return

    for filename in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, filename)
        
        # Handle markdown files
        if filename.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract title from the first H1 heading or use filename
                title = ""
                lines = content.split('\n')
                for line in lines:
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break
                if not title:
                    title = os.path.splitext(filename)[0]
                
                alpaca_record = {
                    "instruction": "Read and internalize the following information about the Sancaktepe municipality to be able to answer questions about it.",
                    "input": title,
                    "output": content
                }
                dataset.append(alpaca_record)

        # Handle JSON files
        elif filename.endswith('.json'):
            if filename == 'url_bank.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        url_data = json.load(f)
                        for item in url_data:
                            # Create a cohesive text block from the JSON object
                            text_block = f"Başlık: {item.get('title', '')}\n"
                            text_block += f"Açıklama: {item.get('description', '')}\n"
                            text_block += f"İçerik: {item.get('content', '')}\n"
                            text_block += f"Kategori: {item.get('category', '')}\n"
                            text_block += f"Anahtar Kelimeler: {', '.join(item.get('keywords', []))}\n"
                            text_block += f"URL: {item.get('url', '')}"
                            
                            alpaca_record = {
                                "instruction": "Read and internalize the following information about a specific web page related to the Sancaktepe municipality.",
                                "input": item.get('title', 'Başlıksız'),
                                "output": text_block
                            }
                            dataset.append(alpaca_record)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {filename}. Skipping.")
                        continue
            else:
                # Generic handling for other JSON files
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        json_data = json.load(f)
                        content = ""
                        if 'text' in json_data:
                            content = json_data['text']
                        else:
                            content = json.dumps(json_data, ensure_ascii=False)
                        
                        alpaca_record = {
                            "instruction": "Read and internalize the following information about the Sancaktepe municipality.",
                            "input": os.path.splitext(filename)[0],
                            "output": content
                        }
                        dataset.append(alpaca_record)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {filename}. Skipping.")
                        continue

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Successfully created '{output_file}' with {len(dataset)} records.")

if __name__ == '__main__':
    create_dataset()