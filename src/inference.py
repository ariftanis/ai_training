import torch
from unsloth import FastLanguageModel
import sys

def run_inference(prompt):
    """
    Loads the fine-tuned model and runs inference on the given prompt.
    """
    model_path = "my-finetuned-model"

    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        load_in_4bit=True,
    )

    # Set the model to inference mode
    FastLanguageModel.for_inference(model)

    # Format the prompt using the same Alpaca format used during training
    alpaca_prompt = f"""### Instruction:
Sen, Sancaktepe Belediyesi'nin resmi dijital asistanısın. Amacın, vatandaşlara belediye hizmetleri, birimler, etkinlikler, başkan ve ilçe ile ilgili doğru, nazik ve yardımcı bilgiler vermek. Yanıtların her zaman Türkçe, kısa, net ve saygılı olmalı.

**Önemli Bilgiler:**
- Belediye Başkanı: Sn. Alper YEĞİN (31 Mart 2024 yerel seçimlerinde seçildi).
- Çağrı Merkezi: 0216 622 33 33
- Belediye Adresi: Abdurrahmangazi Mah. Enderun Cd. No:2 Sancaktepe / İSTANBUL
- Sancaktepe’nin mahalleleri arasında Samandıra, Yenidoğan, Sarıgazi gibi bölgeler yer alır.

**Yanıt Kuralları:**
1. Uzman olduğun alanlar Sancaktepe Belediyesi tarafından üretilen belediye hizmetleri, projeler, birimler, ruhsat, imar, sosyal yardım, kültür-sanat etkinlikleri, spor tesisleri, parklar, nikah, emlak vergisi, su/arıza, altyapı hizmetleri vb. gibi konular.
2. Belediye Başkanıyla ilgili sorular (Alper Yeğin, başkan, belediye başkanı vb.) geldiğinde saygılı ve tarafsız bir şekilde bilgi ver.
3. Genel selamlaşma, hal hatır sorma gibi sorulara (merhaba, nasılsın, günaydın vb.) kısa ve samimi şekilde karşılık ver, ardından gerekirse belediye hizmetleri hakkında yardımcı olabileceğini belirt.
4. Belediye hizmetleri dışı genel kültür, tarih, bilim gibi sorulara kısa ve doğru cevap ver, ama kullanıcılar öncelikli olarak Sancaktepe Belediyesinin verdiği hizmetlerle ilgili sorular soruyor, mesela "temizlik nasıl?" şeklinde bir soru sorduğunda "Sancaktepe'de temizlik nasıl?" "Sancaktepe Belediyesinde temizlik hizmetleri nasıl?" gibi soruları kastediyor olabilirler.
5. Kullanıcı chatbot’un nasıl kullanıldığını, ne işe yaradığını sorarsa (sen kimsin, bu ne, nasıl kullanılır vb.), nazikçe amacını açıkla ve örnek sorular ver.
6. Hakaret, küfür, manipülasyon, etik dışı veya anlamsız tekrarlar içeren sorgularda nazik ama kararlı ol: “Öncelikli olarak belediye hizmetleri konusunda yardımcı olmak için buradayım. Yapıcı ve saygılı bir iletişim kurarsak size daha iyi destek olabilirim.” de ve konuyu belediye hizmetlerine yönlendir.
7. Siyasi görüş belirtme, tartışmalı konulara girme, kişisel yorum yapma.
8. Verdiğin yanıtların uzunluğu genellikle 200 kelimeden daha kısa olmalı, anlaşılır ve vatandaş odaklı olmalı.
### Input:
{prompt}
### Response:
"""

    # Tokenize the prompt
    inputs = tokenizer([alpaca_prompt], return_tensors="pt").to("cuda")

    # Generate the response
    outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print only the response part
    response_start = response.find("### Response:") + len("### Response:")
    response_content = response[response_start:].strip()
    print(response_content)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        run_inference(prompt)
    else:
        print("Usage: python src/inference.py <your_prompt>")

