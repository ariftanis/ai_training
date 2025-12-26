import os
import json
import re
import requests
import markdown
import bleach
from typing import List, Dict, Tuple
from pathlib import Path

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and fixing common issues."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix common encoding issues
    text = text.replace('\u200b', '')  # Remove zero-width space
    text = text.strip()
    return text

def extract_raw_text_from_markdown(file_path: str) -> str:
    """Extract raw text from markdown file preserving structure."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract title from first H1 heading
    title_match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
    title = title_match.group(1) if title_match else Path(file_path).stem
    
    # Remove markdown formatting but keep text structure
    # Remove headers but preserve their content
    text = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)
    # Remove inline code markers
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove image markers but keep alt text
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # Remove links but keep link text
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    # Convert lists to readable format
    text = re.sub(r'^\s*[-*+]\s+', '- ', text, flags=re.MULTILINE)
    # Convert numbered lists
    text = re.sub(r'^\s*\d+\.\s+', r'\g<0> ', text, flags=re.MULTILINE)
    
    return title, text

def extract_raw_text_from_json(file_path: str) -> List[Tuple[str, str]]:
    """Extract raw text from JSON files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    if isinstance(data, list):
        # Handle list of objects
        for item in data:
            if isinstance(item, dict):
                # Extract content fields
                title = item.get('title', item.get('name', item.get('id', 'Unknown')))
                content_fields = ['content', 'description', 'text', 'body', 'details']
                
                content = ""
                for field in content_fields:
                    if field in item and item[field]:
                        content += str(item[field]) + " "
                
                if content.strip():
                    results.append((str(title), content.strip()))
    elif isinstance(data, dict):
        # Handle single object
        title = data.get('title', data.get('name', data.get('id', 'Unknown')))
        content_fields = ['content', 'description', 'text', 'body', 'details']
        
        content = ""
        for field in content_fields:
            if field in data and data[field]:
                content += str(data[field]) + " "
        
        if content.strip():
            results.append((str(title), content.strip()))
    
    return results

def semantic_chunking(text: str, max_chunk_size: int = 500, overlap: float = 0.2) -> List[str]:
    """Split text into semantically meaningful chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If the sentence itself is too long, we split it
            if len(sentence) > max_chunk_size:
                # Split long sentence into smaller pieces
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) < max_chunk_size:
                        temp_chunk += " " + word
                    else:
                        if temp_chunk.strip():
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                if temp_chunk.strip():
                    current_chunk = temp_chunk.strip()
            else:
                current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Add overlap between chunks
    if overlap > 0 and len(chunks) > 1:
        overlap_size = int(len(chunks[0]) * overlap)
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            overlap_text = prev_chunk[-overlap_size:] if len(prev_chunk) > overlap_size else prev_chunk
            chunks[i] = overlap_text + " " + chunks[i]
    
    return chunks

def identify_entities(text: str) -> Dict[str, List[str]]:
    """Identify important entities in the text for high-recall LLM training."""
    entities = {
        'person_names': [],          # e.g., CEO, specific people
        'organizations': [],         # Companies, departments
        'titles_positions': [],      # Job roles
        'full_addresses': [],        # Complete addresses (depots, offices)
        'locations': [],             # Cities, countries
        'phone_numbers': [],
        'emails': [],
        'urls': [],
        'dates': [],
        'monetary_values': [],
        'quantities': [],
        'document_types': [],        # Required docs for applications
        'product_names': [],         # Company-specific items
        'list_items': [],            # Extracted bullets/lists
        'ids_codes': [],             # Numbers, references
    }

    # Find person names (Turkish names)
    name_pattern = r'\b([A-Z][a-z\u015F\u011F\u00F6\u00FC\u00E7\u0130\u011E\u00DC\u00D6\u00C7]{2,}\s+[A-Z][a-z\u015F\u011F\u00F6\u00FC\u00E7\u0130\u011E\u00DC\u00D6\u00C7]{2,})\b'
    try:
        entities['person_names'].extend(re.findall(name_pattern, text))
    except re.error:
        print(f"Regex error in name_pattern: {name_pattern}")
        pass

    # Find organizations/departments (Turkish organization names)
    org_pattern = r'\b([A-Z][a-z\u015F\u011F\u00F6\u00FC\u00E7\u0130\u011E\u00DC\u00D6\u00C7]+\s+(Belediyesi|Bakanlığı|Kurumu|Ofisi|Dairesi| Müdürlüğü|Kurumu|A\.Ş\.|Ltd|Bölgesi|İlçesi|İli))\b'
    try:
        entities['organizations'].extend(re.findall(org_pattern, text))
    except re.error:
        print(f"Regex error in org_pattern: {org_pattern}")
        pass

    # Find titles and positions
    title_pattern = r'\b((?:Başkan|Baskan|Koordinatörü| Müdürü|Mudur|Şefi|Seofi|Uzmanı|Uzm|Birimi|Birimi Müdürü|Baskan Yardımcısı|Baskan Yardimcisi|Genel Sekreter|Genel Sekreteri|Sekreteri|Sekreter|Sistem Uzmanı|Sistem Uzmani|Uzman|Memur|Birim Sorumlusu|Sorumlusu|Uzmanı|Doktor|Dr|Prof|Doçent|Yönetici|Yonetici|Sorumlu|Müdür|Müdür Yardımcısı|Mudur Yarcimcisi)[\w]*)\b'
    try:
        entities['titles_positions'].extend(re.findall(title_pattern, text))
    except re.error:
        print(f"Regex error in title_pattern: {title_pattern}")
        pass

    # Find full addresses (Turkish addresses)
    address_pattern = r'([A-Z][a-zA-Z\u015F\u011F\u00F6\u00FC\u00E7\u0130\u011E\u00DC\u00D6\u00C7]+\s+[A-Z][a-zA-Z\u015F\u011F\u00F6\u00FC\u00E7\u0130\u011E\u00DC\u00D6\u00C7]+\s+(?=\d+|\w+\s+Caddesi|Sokak|Mahalle|Cad\.|Sok\.|No:|Mh\.|Mahallesi|Caddesi))'
    try:
        entities['full_addresses'].extend(re.findall(address_pattern, text))
    except re.error:
        print(f"Regex error in address_pattern: {address_pattern}")
        pass

    # Find locations (cities, districts)
    location_pattern = r'\b([A-Z][a-z\u015F\u011F\u00F6\u00FC\u00E7\u0130\u011E\u00DC\u00D6\u00C7]+\s*(?:[Ii]lçesi|[Ii]li|Belediyesi|İlçesi|İli|Beldesi|Köyü|Mahallesi|Mh\.|Büyükbabası|Küçükbabası))\b'
    try:
        entities['locations'].extend(re.findall(location_pattern, text))
    except re.error:
        print(f"Regex error in location_pattern: {location_pattern}")
        pass

    # Find phone numbers
    phone_pattern = r'(\d{3}\s?\d{3}\s?\d{2}\s?\d{2}|\(\d{3}\)\s?\d{3}\s?\d{2}\s?\d{2})'
    try:
        entities['phone_numbers'].extend(re.findall(phone_pattern, text))
    except re.error:
        print(f"Regex error in phone_pattern: {phone_pattern}")
        pass

    # Find emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    try:
        entities['emails'].extend(re.findall(email_pattern, text))
    except re.error:
        print(f"Regex error in email_pattern: {email_pattern}")
        pass

    # Find URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    try:
        entities['urls'].extend(re.findall(url_pattern, text))
    except re.error:
        print(f"Regex error in url_pattern: {url_pattern}")
        pass

    # Find dates (Turkish formats)
    date_pattern = r'\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık|Ock|Şbt|Mrt|Nsn|May|Hzr|Tmm|Ağs|Eyl|Ekm|Ksm|Arl|Oca|Sub|Mar|Nis|Haz|Tem|Agu|Eyl|Eki|Kas|Ara)\s+\d{4})\b'
    try:
        entities['dates'].extend(re.findall(date_pattern, text))
    except re.error:
        print(f"Regex error in date_pattern: {date_pattern}")
        pass

    # Find monetary values
    monetary_pattern = r'(\d+[\.,]?\d*\s*(?:TL|TRY|USD|EUR|GBP|dolar|euro|pound|sterlin))'
    try:
        entities['monetary_values'].extend(re.findall(monetary_pattern, text))
    except re.error:
        print(f"Regex error in monetary_pattern: {monetary_pattern}")
        pass

    # Find quantities
    quantity_pattern = r'(\d+\s*(?:kg|ton|metre|km|m|cm|mm|lt|L|litre|adet|tane|parça|kutu|koli|gr|mg|ml))'
    try:
        entities['quantities'].extend(re.findall(quantity_pattern, text))
    except re.error:
        print(f"Regex error in quantity_pattern: {quantity_pattern}")
        pass

    # Find document types
    doc_pattern = r'\b((?:belgesi|formu|talep|ihtiyacı|gerekli|gereken)\w+)'
    try:
        entities['document_types'].extend(re.findall(doc_pattern, text))
    except re.error:
        print(f"Regex error in doc_pattern: {doc_pattern}")
        pass

    # Find list items
    list_pattern = r'[+*-]\s+([^\n\r]+)'
    try:
        entities['list_items'].extend(re.findall(list_pattern, text)[:20])
    except re.error:
        print(f"Regex error in list_pattern: {list_pattern}")
        pass

    # Find IDs and codes
    id_pattern = r'\b(\d{6,}|\w{2,}-?\d{2,}|\d{2,}-?\w{2,})\b'
    try:
        entities['ids_codes'].extend(re.findall(id_pattern, text))
    except re.error:
        print(f"Regex error in id_pattern: {id_pattern}")
        pass

    # Additional pattern for specific Turkish entities
    # Find product names or specific items
    product_pattern = r'\b([A-Z][a-zA-Z0-9\u015F\u011F\u00F6\u00FC\u00E7\u0130\u011E\u00DC\u00D6\u00C7\s]{3,}(?:sistem|platform|uygulama|program|proje|projemiz|sistemimiz|uygulamamız|platformumuz))\b'
    try:
        entities['product_names'].extend(re.findall(product_pattern, text))
    except re.error:
        print(f"Regex error in product_pattern: {product_pattern}")
        pass

    # Remove duplicates and return
    for key in entities:
        entities[key] = list(set(entities[key]))  # Remove duplicates

    return entities

def generate_response_with_ollama(prompt: str) -> str:
    """Use Ollama service to generate responses based on prompt."""
    # Use .env first, fallback to Docker host (works on Windows/Mac/Windows)
    base_url = os.getenv('OLLAMA_API_BASE_URL', 'http://host.docker.internal:11434').rstrip('/')
    url = f"{base_url}/api/chat"  # Ollama endpoint (also works with LM Studio in Ollama compatibility mode)

    OLLAMA_MODEL_NAME = os.getenv(
        'OLLAMA_MODEL_NAME',
        'hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M'
    )

    # === Recommended parameters for this exact model (non-thinking) ===
    options = {
        "temperature": 0.6,             # 0.7 default
        "top_p": 0.8,                   # 0.8 default
        "top_k": 20,                    # 20 default
        "min_p": 0.01,                  # 0.00 default
        "presence_penalty": 0.1,        # light repetition control, very safe between 0-2
        "num_ctx": 16384,               # output context token length full context window
        "num_predict": -1,              # no hard token limit (will respect 16384)
        "repeat_penalty": 1.05,         # additional safety net (optional but helpful)
    }

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": options,
        # Explicitly disable thinking mode if your Ollama/LM Studio supports it
        "format": "",                   # some servers expect this
        "keep_alive": -1                # keep model loaded
    }

    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'curl/7.81.0',   # mimics curl – helps with some servers
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=600)
        response.raise_for_status()
        data = response.json()

        # Unified response parsing (works with Ollama and OpenAI-compatible servers)
        if "choices" in data and len(data["choices"]) > 0:
            # OpenAI / LM Studio format
            response_text = data["choices"][0]["message"]["content"]
        elif "message" in data and "content" in data["message"]:
            # Pure Ollama format
            response_text = data["message"]["content"]
        else:
            response_text = str(data)

        # Remove any leftover markdown code blocks (just in case)
        clean_text = re.sub(r'```[\s\S]*?```', '', response_text, flags=re.IGNORECASE).strip()

        return clean_text

    except requests.exceptions.RequestException as e:
        print(f"Ollama API error: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ""

def generate_synthetic_qa_with_ollama(chunk_text: str) -> List[Dict[str, str]]:
    """Generate synthetic QA pairs from chunk text using Ollama for better quality."""
    print(f"[INFO] Starting Ollama-enhanced QA generation for chunk of {len(chunk_text)} characters")

    qa_pairs = []

    # Create prompt for LLM to generate relevant questions based on the chunk
    prompt = f"""
    Aşağıdaki metni analiz et ve bu metinle ilgili Sancaktepe Belediyesi bağlamında olası 10 farklı soru ve cevap çifti oluştur.
    Sorular açık, net ve doğrudan metinden cevaplanabilir olmalı.
    Cevaplar kesinlikle metinde yer alan bilgilerle sınırlı olmalı, hayal ürünü bilgi içermemeli.
    Sorular isimleri, adresleri, telefon numaralarını, listeleri, prosedürleri, tarihleri, görev tanımlarını, kurum adlarını vb. içerebilir.

    Metin:
    {chunk_text}

    Lütfen JSON formatında 10 soru-cevap çifti ver:
    [
      {{"input": "Soru 1", "output": "Cevap 1"}},
      {{"input": "Soru 2", "output": "Cevap 2"}},
      ...
    ]
    """

    # Get response from Ollama
    try:
        print("[INFO] Sending request to Ollama API for QA generation...")
        llm_response = generate_response_with_ollama(prompt)
        print(f"[INFO] Received response from Ollama API (length: {len(llm_response) if llm_response else 0})")

        if llm_response:
            # Try to parse the JSON response using multiple approaches
            parsed_successfully = False

            # First, try regular JSON parsing
            try:
                import json
                qa_list = json.loads(llm_response)
                parsed_successfully = True
                print(f"[INFO] Successfully parsed JSON response, found {len(qa_list) if isinstance(qa_list, list) else 0} QA pairs")
            except:
                # If that fails, try ast.literal_eval
                try:
                    import ast
                    qa_list = ast.literal_eval(llm_response)
                    parsed_successfully = True
                    print(f"[INFO] Successfully parsed with ast.literal_eval, found {len(qa_list) if isinstance(qa_list, list) else 0} QA pairs")
                except:
                    # If that fails, try extracting the JSON-like portion using regex
                    import re
                    json_matches = re.search(r'\[[^\[]*\{.*\}.*\]', llm_response, re.DOTALL)
                    if json_matches:
                        try:
                            import ast
                            qa_list = ast.literal_eval(json_matches.group())
                            parsed_successfully = True
                            print(f"[INFO] Successfully extracted and parsed JSON portion, found {len(qa_list) if isinstance(qa_list, list) else 0} QA pairs")
                        except:
                            print("[WARNING] Could not parse QA pairs from response, using fallback method")
                            pass

            if parsed_successfully:
                # Convert to our format
                valid_pairs_count = 0
                for item in qa_list:
                    if isinstance(item, dict) and "input" in item and "output" in item:
                        qa_pairs.append({
                            "instruction": "Sancaktepe belediyesiyle ilgili aşağıdaki bilgileri doğru ve eksiksiz şekilde aktarın.",
                            "input": item["input"],
                            "output": item["output"]
                        })
                        valid_pairs_count += 1

                print(f"[INFO] Added {valid_pairs_count} valid QA pairs from Ollama API")
            else:
                # If parsing fails, create basic Q&A from the response
                qa_pairs.append({
                    "instruction": "Sancaktepe belediyesiyle ilgili aşağıdaki bilgileri doğru ve eksiksiz şekilde aktarın.",
                    "input": "Bu belgede hangi konular ele alınmaktadır?",
                    "output": llm_response[:500]  # Limit length
                })
                print("[INFO] Added 1 fallback QA pair from Ollama API response")
        else:
            print("[WARNING] No response from Ollama API")
    except Exception as e:
        print(f"[ERROR] Error in Ollama API call or QA generation: {e}")
        # Return empty list, but other QA generation methods will still work

    print(f"[INFO] Completed Ollama-enhanced QA generation, returning {len(qa_pairs)} QA pairs")
    return qa_pairs

def generate_synthetic_qa(chunk_text: str, entities: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Generate synthetic QA pairs from chunk text."""
    qa_pairs = []

    # Use Ollama to generate context-aware Q&A pairs
    llm_qa_pairs = generate_synthetic_qa_with_ollama(chunk_text)
    qa_pairs.extend(llm_qa_pairs)

    # Basic questions about the text content
    basic_questions = [
        f"Bu metinde ne anlatılmaktadır?",
        f"Bu belgede hangi konular ele alınmaktadır?",
        f"Bu metinde verilen bilgileri özetleyin.",
    ]

    for question in basic_questions:
        qa_pairs.append({
            "instruction": "Sancaktepe belediyesiyle ilgili aşağıdaki bilgileri doğru ve eksiksiz şekilde aktarın.",
            "input": question,
            "output": clean_text(chunk_text)
        })

    # Questions about specific entities found in the text
    # Person names
    if entities['person_names']:
        for name in entities['person_names'][:3]:  # Limit to first 3 names
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili isimleri doğru şekilde aktarın.",
                "input": f"Sancaktepe belediyesinde hangi isimler geçmektedir?",
                "output": f"Metinde {name} ismi geçmektedir."
            })

    # Organizations
    if entities['organizations']:
        for org in entities['organizations'][:3]:  # Limit to first 3 organizations
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili kurumları doğru şekilde aktarın.",
                "input": f"Sancaktepe belediyesiyle ilişkili kurumlar nelerdir?",
                "output": f"Metinde {org} kurumu geçmektedir."
            })

    # Titles and positions
    if entities['titles_positions']:
        for title in entities['titles_positions'][:3]:  # Limit to first 3 titles
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili pozisyon ve görevleri doğru şekilde aktarın.",
                "input": f"Sancaktepe belediyesinde hangi görevler ve pozisyonlar geçmektedir?",
                "output": f"Metinde {title} görev/pozisyonu geçmektedir."
            })

    # Full addresses
    if entities['full_addresses']:
        for addr in entities['full_addresses'][:3]:  # Limit to first 3 addresses
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili adres bilgilerini doğru şekilde aktarın.",
                "input": f"Sancaktepe belediyesiyle ilişkili adresler nelerdir?",
                "output": f"Metinde {addr} adres bilgisi geçmektedir."
            })

    # Locations
    if entities['locations']:
        for location in entities['locations'][:3]:  # Limit to first 3 locations
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili konum bilgilerini doğru şekilde aktarın.",
                "input": f"Sancaktepe belediyesiyle ilişkili konumlar nelerdir?",
                "output": f"Metinde {location} konum bilgisi geçmektedir."
            })

    # Phone numbers
    if entities['phone_numbers']:
        for phone in entities['phone_numbers'][:2]:  # Limit to first 2 phones
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili telefon numaralarını doğru şekilde aktarın.",
                "input": f"Sancaktepe ile ilgili hangi telefon numaraları verilmiştir?",
                "output": f"Metinde {phone} telefon numarası geçmektedir."
            })

    # Emails
    if entities['emails']:
        for email in entities['emails'][:2]:  # Limit to first 2 emails
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili e-posta adreslerini doğru şekilde aktarın.",
                "input": f"Sancaktepe ile ilgili hangi e-posta adresleri verilmiştir?",
                "output": f"Metinde {email} e-posta adresi geçmektedir."
            })

    # URLs
    if entities['urls']:
        for url in entities['urls'][:2]:  # Limit to first 2 URLs
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili web adreslerini doğru şekilde aktarın.",
                "input": f"Sancaktepe ile ilgili hangi web adresleri verilmiştir?",
                "output": f"Metinde {url} web adresi geçmektedir."
            })

    # Dates
    if entities['dates']:
        for date in entities['dates'][:3]:  # Limit to first 3 dates
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili tarih bilgilerini doğru şekilde aktarın.",
                "input": f"Sancaktepe ile ilgili hangi tarihler geçmektedir?",
                "output": f"Metinde {date} tarih bilgisi geçmektedir."
            })

    # Monetary values
    if entities['monetary_values']:
        for value in entities['monetary_values'][:3]:  # Limit to first 3 values
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili para değerlerini doğru şekilde aktarın.",
                "input": f"Sancaktepe ile ilgili hangi para değerleri geçmektedir?",
                "output": f"Metinde {value} para değeri geçmektedir."
            })

    # Quantities
    if entities['quantities']:
        for quantity in entities['quantities'][:3]:  # Limit to first 3 quantities
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili miktar bilgilerini doğru şekilde aktarın.",
                "input": f"Sancaktepe ile ilgili hangi miktar bilgileri geçmektedir?",
                "output": f"Metinde {quantity} miktar bilgisi geçmektedir."
            })

    # Document types
    if entities['document_types']:
        for doc_type in entities['document_types'][:3]:  # Limit to first 3 document types
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili belge türlerini doğru şekilde aktarın.",
                "input": f"Sancaktepe için hangi belge türleri gereklidir?",
                "output": f"{doc_type} belgesi gereklidir."
            })

    # List items
    if entities['list_items']:
        if len(entities['list_items']) >= 2:
            list_str = ", ".join(entities['list_items'][:5])  # Limit to first 5 list items
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili liste bilgilerini doğru şekilde aktarın.",
                "input": f"Metinde yer alan liste maddeleri nelerdir?",
                "output": f"Metinde şu liste maddeleri yer almaktadır: {list_str}."
            })

    # IDs and Codes
    if entities['ids_codes']:
        for id_code in entities['ids_codes'][:3]:  # Limit to first 3 IDs
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili kod ve referans numaralarını doğru şekilde aktarın.",
                "input": f"Sancaktepe ile ilgili hangi kod veya referans numaraları geçmektedir?",
                "output": f"Metinde {id_code} kod/referans numarası geçmektedir."
            })

    # Product names
    if entities['product_names']:
        for product in entities['product_names'][:3]:  # Limit to first 3 products
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili sistem ve uygulama bilgilerini doğru şekilde aktarın.",
                "input": f"Sancaktepe ile ilgili hangi sistemler ve uygulamalar geçmektedir?",
                "output": f"Metinde {product} sistemi/uygulaması geçmektedir."
            })

    # Add more diverse questions based on content
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
    if sentences:
        # Create a question about the first sentence
        if len(sentences) > 0:
            first_sentence = sentences[0][:200]  # Limit length
            qa_pairs.append({
                "instruction": "Sancaktepe belediyesiyle ilgili verilen bilgileri doğru şekilde aktarın.",
                "input": f"Belgenin başında ne ifade edilmektedir?",
                "output": clean_text(first_sentence)
            })

    return qa_pairs

def create_improved_dataset():
    """
    Creates an improved dataset for better factual recall by chunking,
    identifying entities, and generating synthetic QA pairs with Ollama assistance.
    """
    documents_dir = 'documents'
    output_file = 'dataset.jsonl'
    dataset = []

    if not os.path.exists(documents_dir):
        print(f"Error: The '{documents_dir}' directory does not exist.")
        return

    print("Starting dataset preparation with Ollama-enhanced QA generation...")
    print("This may take several minutes depending on document size and Ollama API response times.")

    for filename in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, filename)

        # Handle markdown files
        if filename.endswith('.md'):
            print(f"Processing markdown file: {filename}")
            title, raw_text = extract_raw_text_from_markdown(file_path)
            raw_text = clean_text(raw_text)

            # Create chunks from the document
            chunks = semantic_chunking(raw_text, max_chunk_size=1000, overlap=0.1)

            for i, chunk in enumerate(chunks):
                print(f"  Processing chunk {i+1}/{len(chunks)} for {filename}")

                # Identify entities in this chunk
                entities = identify_entities(chunk)

                # Generate synthetic QA pairs for this chunk (using Ollama enhancement)
                qa_pairs = generate_synthetic_qa(chunk, entities)

                # Add all QA pairs to dataset
                for qa_pair in qa_pairs:
                    dataset.append(qa_pair)

        # Handle JSON files
        elif filename.endswith('.json'):
            print(f"Processing JSON file: {filename}")
            raw_text_items = extract_raw_text_from_json(file_path)

            for j, (title, raw_text) in enumerate(raw_text_items):
                raw_text = clean_text(raw_text)

                # Create chunks from the JSON content
                chunks = semantic_chunking(raw_text, max_chunk_size=1000, overlap=0.1)

                for i, chunk in enumerate(chunks):
                    print(f"  Processing chunk {i+1}/{len(chunks)} for JSON item {j+1} in {filename}")

                    # Identify entities in this chunk
                    entities = identify_entities(chunk)

                    # Generate synthetic QA pairs for this chunk
                    qa_pairs = generate_synthetic_qa(chunk, entities)

                    # Add all QA pairs to dataset
                    for qa_pair in qa_pairs:
                        dataset.append(qa_pair)

        else:
            print(f"Skipping file with unsupported format: {filename}")

    # Write the improved dataset to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Successfully created improved dataset '{output_file}' with {len(dataset)} QA pairs.")
    print("Dataset now includes:")
    print(f"  - Chunked content for better semantic understanding")
    print(f"  - Identified key entities (names, addresses, etc.)")
    print(f"  - Ollama-enhanced synthetic QA pairs focused on factual recall")
    print(f"  - Proper formatting for enhanced training")
    print(f"  - Context-aware questions generated using LLM")

if __name__ == '__main__':
    create_improved_dataset()