"""
Shakespeare Veri Seti Modülü

Bu modül Tiny Shakespeare veri setinin indirilmesini, yüklenmesini ve hazırlanmasını yönetir.
Basitlik için karakter seviyesinde tokenization kullanıyoruz.

Temel Kavramlar:
1. Tokenization - Karakterleri sayılara dönüştürme
2. Train/Validation Split - Veriyi eğitim ve doğrulama için ayırma
3. Batch Generation - Eğitim için input-target çiftleri oluşturma
"""

import os
import urllib.request
import torch

# Tiny Shakespeare veri setinin URL'si (Andrej Karpathy'den alınmıştır)
SHAKESPEARE_URL = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt"
DATA_PATH = "data/shakespeare.txt"


def download_shakespeare():
    """Shakespeare veri seti mevcut değilse indir."""
    if os.path.exists(DATA_PATH):
        print(f"Dataset already exists at {DATA_PATH}")
        return

    print("Downloading Shakespeare dataset...")
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    print(f"Downloaded to {DATA_PATH}")


class CharacterTokenizer:
    """
    Karakter Seviyesinde Tokenizer

    Metni sayılara ve sayıları tekrar metne dönüştürür.
    Her benzersiz karaktere benzersiz bir ID verilir.

    Örnek:
        'hello' -> [7, 4, 11, 11, 14]
        [7, 4, 11, 11, 14] -> 'hello'

    -------------------------------------------------------------------------
    NEDEN KARAKTER SEVİYESİNDE TOKENIZATION?
    -------------------------------------------------------------------------
    Eğitim açısından daha basit olduğu için karakter seviyesinde tokenization
    kullanıyoruz.

    Endüstride kullanılan modeller (GPT, BERT), BPE ve WordPiece gibi
    subword tokenization yöntemlerini kullanır.
    Bunlar daha verimlidir ancak sisteme ek karmaşıklık getirir.

    ÖNEMLİ: Verdiğimiz integer ID'ler KEYFİDİR; bunlar sadece index'tir!

    Neural network bu sayıları doğrudan "görmez". Bunun yerine:

        Karakter → Integer ID → Embedding Layer → Öğrenilmiş Vektör
           'h'   →     7      →   lookup[7]    → [0.23, -0.45, ...]

    Embedding layer (nn.Embedding) ÖĞRENİLEBİLİR bir lookup table'dır.

    Eğitim sırasında backpropagation bu vektörleri günceller.
    Böylece benzer bağlamlarda bulunan karakterler benzer temsiller öğrenebilir.

    Bu, Word2Vec ile aynı temel prensiptir:
    Gerçek "anlam" keyfi olarak verilen ID'nin içinde değil,
    öğrenilmiş embedding vektörünün içindedir.
    -------------------------------------------------------------------------
    """

    def __init__(self, text: str):
        # 1. Tüm benzersiz karakterleri al ve sırala
        self.characters = sorted(list(set(text)))

        # 2. Vocabulary boyutunu sakla
        self.vocab_size = len(self.characters)

        # 3. Karakterden ID'ye eşleme oluştur
        self.char_to_id = {}
        for index, char in enumerate(self.characters):
            self.char_to_id[char] = index

        # 4. ID'den karaktere eşleme oluştur
        self.id_to_char = {}
        for index, char in enumerate(self.characters):
            self.id_to_char[index] = char

        # 5. Vocabulary bilgilerini yazdır
        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"Characters: {repr(''.join(self.characters[:50]))}...")

    def encode(self, text: str) -> list:
        """Metni integer ID listesinə dönüştür."""
        ids = []
        for char in text:
            ids.append(self.char_to_id[char])
        return ids

    def decode(self, ids: list) -> str:
        """Integer ID listesini tekrar metne dönüştür."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        characters = []
        for id in ids:
            characters.append(self.id_to_char[id])
        return ''.join(characters)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    """
    Rastgele bir eğitim batch'i oluştur.

    -------------------------------------------------------------------------
    LANGUAGE MODEL EĞİTİMİ NASIL ÇALIŞIR?
    -------------------------------------------------------------------------
    TARGET = INPUT'un 1 pozisyon kaydırılmış hali olacak şekilde
    input-target çiftleri oluşturuyoruz.

    Model her pozisyonda BİR SONRAKİ karakteri tahmin etmeyi öğrenir.

    block_size=5 için örnek:

        Metin: "To be or not to be"

        1. Rastgele bir başlangıç pozisyonu seç ve
           (block_size + 1) karakter al:

           chunk = ['T', 'o', ' ', 'b', 'e', ' ']   (6 karakter)

        2. Bunu input (x) ve target (y) olarak ayır:

           x = ['T', 'o', ' ', 'b', 'e']     (ilk 5 karakter)
           y = ['o', ' ', 'b', 'e', ' ']     (son 5 karakter = 1 kaydırılmış)

        3. Model şunları tahmin etmeyi öğrenir:

           'T' verildiğinde       → 'o' tahmin et
           'To' verildiğinde      → ' ' tahmin et
           'To ' verildiğinde     → 'b' tahmin et
           'To b' verildiğinde    → 'e' tahmin et
           'To be' verildiğinde   → ' ' tahmin et

    Bu işlem aynı anda birden fazla rastgele pozisyon
    (batch_size kadar) için yapılır.
    -------------------------------------------------------------------------

    Argümanlar:
        data: Tüm token ID'lerini içeren Tensor
        block_size: Her sequence'in uzunluğu
        batch_size: Her batch içerisindeki sequence sayısı

    Döndürür:
        x: Input sequence'leri, shape (batch_size, block_size)
        y: Target sequence'leri, shape (batch_size, block_size)
    """

    # 1. Rastgele başlangıç pozisyonları seç
    max_start = len(data) - block_size - 1
    positions = torch.randint(max_start, (batch_size,))

    # 2. Her pozisyon için input (x) ve target (y) oluştur
    x_list = []
    y_list = []

    for pos in positions:
        x_list.append(data[pos : pos + block_size])        # Input: 0'dan n-1'e kadar olan karakterler
        y_list.append(data[pos + 1 : pos + block_size + 1])  # Target: 1'den n'e kadar olan karakterler (1 kaydırılmış)

    # 3. Tensor'ları batch haline getir: (batch_size, block_size)
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x, y


def load_data(block_size: int = 256, train_split: float = 0.9):
    """
    Shakespeare veri setini yükle ve hazırla.

    Argümanlar:
        block_size: Her sequence'in uzunluğu
        train_split: Verinin eğitim için kullanılacak oranı
                     (geri kalanı validation için kullanılır)

    Döndürür:
        train_data: Tensor olarak eğitim token ID'leri
        val_data: Tensor olarak validation token ID'leri
        tokenizer: Character tokenizer
    """

    # 1. Veri setini indir
    download_shakespeare()

    # 2. Metin dosyasını yükle
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        text = file.read()

    print(f"\nDataset size: {len(text):,} characters")
    print(f"Sample text:\n{text[:200]}")
    print("..." + "-" * 50)

    # 3. Tokenizer'ı oluştur
    tokenizer = CharacterTokenizer(text)

    # 4. Tüm metni token ID'lerine encode et
    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids, dtype=torch.long)

    # 5. Veriyi training ve validation setlerine ayır
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

    # 2. Örnek bir batch oluştur
    x, y = get_batch(train_data, block_size=128, batch_size=4)

    print(f"\nSample batch:")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")

    # 3. Input-target ilişkisini göster
    print(f"\n--- Demonstrating input-target pairs ---")
    print(f"Input (x[0]):  {tokenizer.decode(x[0][:20])}...")
    print(f"Target (y[0]): {tokenizer.decode(y[0][:20])}...")
    print(f"Notice: target is input shifted by 1 character!")