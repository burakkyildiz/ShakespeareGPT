"""
Shakespeare Veri Seti Modülü

Bu modül Tiny Shakespeare veri setini yükler ve hazırlar.
Basitlik için karakter bazlı tokenization kullanıyoruz.

Temel Kavramlar:
1. Tokenization - Karakterleri sayılara çevirme
2. Train/Validation Ayrımı - Veriyi eğitim ve değerlendirme için ayırma
3. Batch Üretimi - Eğitim için girdi-hedef çiftleri oluşturma
"""

import os
import urllib.request
import torch

# Tiny Shakespeare veri setinin URL'si (Andrej Karpathy'den fork'lanmış)
SHAKESPEARE_URL = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt"
DATA_PATH = "data/shakespeare.txt"


def download_shakespeare():
    """Shakespeare veri setini, daha önce indirilmediyse indirir."""
    if os.path.exists(DATA_PATH):
        print(f"Dataset already exists at {DATA_PATH}")
        return

    print("Downloading Shakespeare dataset...")
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    print(f"Downloaded to {DATA_PATH}")


class CharacterTokenizer:
    """
    Karakter Bazlı Tokenizer

    Metni sayılara, sayıları da tekrar metne çevirir.
    Her benzersiz karakter, benzersiz bir ID alır.

    Örnek:
        'hello' -> [7, 4, 11, 11, 14]
        [7, 4, 11, 11, 14] -> 'hello'

    -------------------------------------------------------------------------
    NEDEN KARAKTER BAZLI TOKENIZATION?
    -------------------------------------------------------------------------
    Eğitim açısından basit olduğu için karakter bazlı tokenization kullanıyoruz.
    Endüstri modelleri (GPT, BERT) alt kelime (subword) tokenization kullanır
    (BPE, WordPiece). Bu daha verimlidir ama karmaşıklık ekler.

    ÖNEMLİ: Atadığımız tam sayı ID'leri KEYFİDİR, sadece birer indekstir!
    Sinir ağı bu sayıları doğrudan "görmez". Bunun yerine:

        Karakter → Tam Sayı ID → Embedding Katmanı → Öğrenilmiş Vektör
           'h'   →      7      →    lookup[7]      → [0.23, -0.45, ...]

    Embedding katmanı (nn.Embedding) ÖĞRENİLEBİLİR bir arama tablosudur.
    Eğitim sırasında backpropagation bu vektörleri günceller; böylece
    benzer bağlamlarda geçen karakterler benzer temsiller kazanır.

    Bu, Word2Vec ile aynı prensiptir: asıl "anlam" keyfi ID'de değil,
    öğrenilmiş embedding vektöründedir.
    -------------------------------------------------------------------------
    """

    def __init__(self, text: str):
        # 1. Tüm benzersiz karakterleri al ve sırala
        self.characters = sorted(list(set(text)))

        # 2. Sözlük boyutunu sakla
        self.vocab_size = len(self.characters)

        # 3. Karakter -> ID eşlemesini oluştur
        self.char_to_id = {}
        for index, char in enumerate(self.characters):
            self.char_to_id[char] = index

        # 4. ID -> karakter eşlemesini oluştur
        self.id_to_char = {}
        for index, char in enumerate(self.characters):
            self.id_to_char[index] = char

        # 5. Sözlük bilgisini yazdır
        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"Characters: {repr(''.join(self.characters[:50]))}...")

    def encode(self, text: str) -> list:
        """Metni tam sayı ID'lerinden oluşan bir listeye çevirir."""
        ids = []
        for char in text:
            ids.append(self.char_to_id[char])
        return ids

    def decode(self, ids: list) -> str:
        """Tam sayı ID listesini tekrar metne çevirir."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        characters = []
        for id in ids:
            characters.append(self.id_to_char[id])
        return ''.join(characters)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    """
    Rastgele bir eğitim verisi batch'i (grubu) üretir.

    -------------------------------------------------------------------------
    DİL MODELİ EĞİTİMİ NASIL ÇALIŞIR
    -------------------------------------------------------------------------
    Girdi-hedef çiftleri oluştururuz. HEDEF = GİRDİ'nin 1 pozisyon kaydırılmış hali.
    Model, her pozisyonda bir SONRAKİ karakteri tahmin etmeyi öğrenir.

    block_size=5 ile örnek:

        Metin: "To be or not to be"

        1. Rastgele bir başlangıç pozisyonu seç, (block_size + 1) karakter al:
           chunk = ['T', 'o', ' ', 'b', 'e', ' ']   (6 karakter)

        2. Girdi (x) ve hedef (y) olarak ayır:
           x = ['T', 'o', ' ', 'b', 'e']     (ilk 5 karakter)
           y = ['o', ' ', 'b', 'e', ' ']     (son 5 karakter = 1 kaydırılmış)

        3. Model şunları tahmin etmeyi öğrenir:
           'T' verilince       → 'o' tahmin et
           'To' verilince      → ' ' tahmin et
           'To ' verilince     → 'b' tahmin et
           'To b' verilince    → 'e' tahmin et
           'To be' verilince   → ' ' tahmin et

    Bu işlem, birden fazla rastgele pozisyon için (batch_size kadar) aynı anda yapılır.
    -------------------------------------------------------------------------

    Args:
        data: Tüm token ID'lerini içeren tensör
        block_size: Her dizinin uzunluğu
        batch_size: Bir batch'teki dizi sayısı

    Returns:
        x: Girdi dizileri, şekil (batch_size, block_size)
        y: Hedef dizileri, şekil (batch_size, block_size)
    """
    # 1. Rastgele başlangıç pozisyonları seç
    max_start = len(data) - block_size - 1
    positions = torch.randint(max_start, (batch_size,))

    # 2. Her pozisyon için girdi (x) ve hedef (y) çıkar
    x_list = []
    y_list = []

    for pos in positions:
        x_list.append(data[pos : pos + block_size])        # Girdi: 0'dan n-1'e kadar karakterler
        y_list.append(data[pos + 1 : pos + block_size + 1])  # Hedef: 1'den n'e kadar karakterler (1 kaydırılmış)

    # 3. Batch tensörlerine birleştir (stack): (batch_size, block_size)
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x, y


def load_data(block_size: int = 256, train_split: float = 0.9):
    """
    Shakespeare veri setini yükler ve hazırlar.

    Args:
        block_size: Her dizinin uzunluğu
        train_split: Eğitim için ayrılan veri oranı (kalanı validation olur)

    Returns:
        train_data: Eğitim token ID'leri (tensör)
        val_data: Validation token ID'leri (tensör)
        tokenizer: Karakter tokenizer'ı
    """
    # 1. Veri setini indir
    download_shakespeare()

    # 2. Metin dosyasını oku
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        text = file.read()

    print(f"\nDataset size: {len(text):,} characters")
    print(f"Sample text:\n{text[:200]}")
    print("..." + "-" * 50)

    # 3. Tokenizer'ı oluştur
    tokenizer = CharacterTokenizer(text)

    # 4. Tüm metni token ID'lerine çevir
    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids, dtype=torch.long)

    # 5. Eğitim ve validation setlerine ayır
    split_index = int(train_split * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    print(f"\nTrain size: {len(train_data):,} tokens")
    print(f"Val size: {len(val_data):,} tokens")

    return train_data, val_data, tokenizer


# Veri setini test et
if __name__ == "__main__":
    # 1. Veriyi yükle
    train_data, val_data, tokenizer = load_data(block_size=128)

    # 2. Örnek bir batch al
    x, y = get_batch(train_data, block_size=128, batch_size=4)

    #print(f"\nSample batch:")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")

    # 3. Girdi-hedef ilişkisini göster
    #print(f"\n--- Demonstrating input-target pairs ---")
    #print(f"Input (x[0]):  {tokenizer.decode(x[0][:20])}")
    #print(f"Target (y[0]): {tokenizer.decode(y[0][:20])}")
    #print(f"Notice: target is input shifted by 1 character!")