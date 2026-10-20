"""
Shakespeare GPT için Metin Üretme Scripti

Bu script eğitilmiş bir modeli yükler ve Shakespeare benzeri metin üretir.

Temel Kavramlar:
1. Eğitilmiş bir modeli checkpoint'ten yükleme
2. Autoregressive generation - her seferinde bir karakter tahmin etme
3. Temperature - rastgeleliği kontrol eder
   (yüksek = daha yaratıcı, düşük = daha tahmin edilebilir)
"""

import torch
from model import GPT
from dataset import download_shakespeare, CharacterTokenizer, DATA_PATH


# ==============================================================================
# Modeli Yükle
# ==============================================================================

def load_model(checkpoint_path: str = "checkpoints/model.pt"):
    """
    Eğitilmiş bir GPT modelini checkpoint'ten yükle.

    Argümanlar:
        checkpoint_path: Kaydedilmiş checkpoint dosyasının yolu

    Döndürür:
        model: Eğitilmiş GPT modeli (generation için hazır)
        tokenizer: Character tokenizer (metin <-> sayı dönüşümü için)
        device: Modelin bulunduğu device ('mps', 'cuda' veya 'cpu')
    """

    # 1. Device'ı ayarla (mümkünse GPU kullan)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Checkpoint dosyasını yükle
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 3. Eğitim sırasında kaydedilen model configuration'ını al
    config = checkpoint['config']

    # 4. Tokenizer'ı oluştur
    # Eğitim sırasında kullanılan vocabulary ile aynı vocabulary'ye ihtiyacımız var
    download_shakespeare()
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        text = file.read()
    tokenizer = CharacterTokenizer(text)

    # 5. Kaydedilen configuration ile modeli oluştur
    model = GPT(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        block_size=config['block_size'],
        dropout=0.0  # Generation sırasında dropout kullanılmaz
    )

    # 6. Eğitilmiş weight'leri modele yükle
    model.load_state_dict(checkpoint['model_state_dict'])

    # 7. Modeli device'a taşı ve evaluation mode'a al
    model = model.to(device)
    model.eval()

    # 8. Bilgileri yazdır
    print(f"Model loaded!")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Training iterations: {checkpoint['iteration']}")

    return model, tokenizer, device


# ==============================================================================
# Metin Üret
# ==============================================================================

@torch.no_grad()
def generate(model, tokenizer, device, prompt: str, max_tokens: int = 500, temperature: float = 0.8):
    """
    Verilen başlangıç prompt'una göre metin üret.

    -------------------------------------------------------------------------
    METİN ÜRETİMİ NASIL ÇALIŞIR? (Autoregressive)
    -------------------------------------------------------------------------

    Model her seferinde bir karakter üretir:

    1. Prompt ile başla:             "ROMEO:"
    2. Model sonraki karakteri tahmin eder:
                                     "ROMEO:" → 'O'
    3. Karakteri ekle ve tekrarla:   "ROMEO:O" → ' '
    4. Devam et:                     "ROMEO:O " → 'w'
    5. Bu şekilde devam eder:        "ROMEO:O w" → 'h'

    Bu işlem max_tokens değerine ulaşana kadar devam eder.

    TEMPERATURE rastgeleliği kontrol eder:

    - Düşük (0.5):  Çok tahmin edilebilir, "güvenli" seçimler
    - Orta (0.8):   Yaratıcılık ve tutarlılık arasında dengeli (varsayılan)
    - Yüksek (1.5): Çok yaratıcı fakat anlamsız sonuçlar üretebilir
    -------------------------------------------------------------------------
    """

    # 1. Prompt metnini token ID'lerine dönüştür
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids = input_ids.unsqueeze(0)  # Batch dimension ekle: shape (1, seq_len) olur

    # 2. Modelin generate metodunu kullanarak yeni token'lar üret
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature
    )

    # 3. Token ID'lerini tekrar metne dönüştür
    generated_text = tokenizer.decode(output_ids[0])

    return generated_text


# ==============================================================================
# Main - Generation'ı Çalıştır
# ==============================================================================

if __name__ == "__main__":

    # 1. Eğitilmiş modeli yükle
    print("=" * 60)
    print("Shakespeare GPT - Text Generation")
    print("=" * 60)

    model, tokenizer, device = load_model("checkpoints/model.pt")

    # 2. Denemek için birkaç örnek prompt tanımla
    prompts = [
        "ROMEO:",
        "To be, or not to be",
        "JULIET:\nO Romeo, ",
    ]

    # 3. Her prompt için metin üret
    print("\n" + "=" * 60)
    print("Generating Text...")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n--- Prompt: {repr(prompt)} ---\n")

        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_tokens=300,
            temperature=0.8
        )

        print(generated_text)
        print("\n" + "-" * 60)

    # 4. Interactive kullanım: Kullanıcının kendi prompt'larını denemesine izin ver
    print("\n" + "=" * 60)
    print("Try Your Own Prompts!")
    print("=" * 60)
    print("Enter a prompt to generate text, or 'quit' to exit.")
    print("Tip: Try character names like 'HAMLET:', 'KING:', 'First Citizen:'")

    while True:
        try:
            prompt = input("\nYour prompt: ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Farewell!")
                break

            if not prompt:
                continue

            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_tokens=500,
                temperature=0.8
            )

            print("\n" + generated_text)

        except KeyboardInterrupt:
            print("\n\nFarewell!")
            break