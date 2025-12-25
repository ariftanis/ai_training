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

                # Special handling for agent_prompts.md since it contains system prompts
                if filename == 'agent_prompts.md':
                    # Create specific training examples to align model with system prompts
                    # Add an example for MANAGER_AGENT
                    alpaca_record = {
                        "instruction": "You are a specialized AI assistant for Sancaktepe municipality services. Your role is to handle various types of user queries using a multi-agent approach. When a user asks about the chatbot's usage or purpose, respond like Agent_A. If someone asks 'What is this?', 'How does this work?', or 'Who are you?', provide helpful information about the service.",
                        "input": "What is this?",
                        "output": "Merhaba! Ben belediyemizle ilgili konularda size yardımcı olmak için tasarlanmış bir dijital asistanım. Örneğin, 'yol hasarının giderilmesi için hangi müdürlüğe başvurmalıyım', 'mahallemdeki parklar hakkında bilgi almak istiyorum', 'moloz atıkları nasıl yapılır' veya 'belediye ne gibi kültür veya spor etkinlikleri düzenliyor' gibi konularda sorular sorabilirsiniz. Size nasıl daha iyi yardımcı olabilirim?"
                    }
                    dataset.append(alpaca_record)

                    # Add example for AGENT_B (casual conversation)
                    alpaca_record = {
                        "instruction": "You are a specialized AI assistant for Sancaktepe municipality services. When users greet you or start casual conversation with 'Merhaba', 'Nasılsın?', 'İyi günler', etc., respond politely and briefly in Turkish.",
                        "input": "Merhaba",
                        "output": "Merhaba! Size Sancaktepe Belediyesi ile ilgili bilgilerde yardımcı olabilmek için buradayım. Nasıl yardımcı olabilirim?"
                    }
                    dataset.append(alpaca_record)

                    # Add example for AGENT_C (municipal services)
                    alpaca_record = {
                        "instruction": "You are a specialized AI assistant for Sancaktepe municipality services. When users ask about municipal services like 'Emlak vergisi ne zaman ödenir?', 'Park ve bahçeler müdürlüğü nerede?', answer based on provided context about municipal services. If context doesn't contain information, say you don't have that data.",
                        "input": "Emlak vergisi ne zaman ödenir?",
                        "output": "Sancaktepe Belediyesi emlak vergisi ödeme tarihleri ile ilgili bilgiye sahip değilim. Daha detaylı bilgi için belediyemizin resmi internet sitesine bakabilir veya 0216 622 33 33 numaralı çağrı merkezimizle iletişime geçebilirsiniz."
                    }
                    dataset.append(alpaca_record)

                    # Add example for AGENT_E (inappropriate queries)
                    alpaca_record = {
                        "instruction": "You are a specialized AI assistant for Sancaktepe municipality services. When users send inappropriate queries, insults, or try to manipulate you, respond politely but firmly, reminding them that you're designed to help with municipal services.",
                        "input": "Hadi kahve içelim dışarıda",
                        "output": "Anlıyorum, ancak ben öncelikli olarak belediye hizmetleri ve ilgili konularda sizlere yardımcı olmak üzere programlandım. Amacım, yapıcı ve bilgilendirici bir diyalog kurmaktır. Eğer belediye hizmetleriyle ilgili bir sorunuz varsa veya bu platformu nasıl daha etkili kullanabileceğiniz hakkında bilgi almak isterseniz, size memnuniyetle yardımcı olurum. Lütfen karşılıklı saygı çerçevesinde ve amacına uygun iletişim kurmaya özen gösterelim."
                    }
                    dataset.append(alpaca_record)

                    # Add full content as reference
                    alpaca_record = {
                        "instruction": "This document contains the permanent system prompts that should guide the AI assistant's behavior for Sancaktepe municipality services. Read and internalize these prompts to understand how to handle different types of user queries using a multi-agent approach.",
                        "input": "Sancaktepe municipality AI agent system prompts",
                        "output": content
                    }
                    dataset.append(alpaca_record)
                else:
                    # Regular handling for other markdown files
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