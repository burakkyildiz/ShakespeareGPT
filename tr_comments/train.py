"""
Shakespeare GPT için Eğitim Scripti

Bu script GPT modelimizi Shakespeare metni üzerinde eğitir.
Eğitim bir MacBook üzerinde yaklaşık 15-30 dakika sürer.

Temel Kavramlar:
1. Training Loop - Forward pass, loss hesaplama, backward pass, optimizer step
2. Learning Rate Warmup - Stabil eğitim için başlangıçta LR'yi kademeli olarak artırma
3. Evaluation - Modelin öğrenip öğrenmediğini kontrol etmek için periyodik validation
"""

import os
import time
import torch
from model import GPT
from dataset import load_data, get_batch


# ==============================================================================
# Hyperparameter'lar (Ayarlar)
# ==============================================================================

# Model mimarisi
EMBEDDING_DIM = 384    # Embedding boyutu (her vektörün ne kadar büyük olduğu)
NUM_HEADS = 6          # Attention head sayısı
NUM_LAYERS = 6         # Transformer block sayısı
BLOCK_SIZE = 256       # Maksimum sequence uzunluğu
DROPOUT = 0.1          # Regularization için dropout oranı

# Eğitim ayarları
BATCH_SIZE = 64        # Her batch'teki sequence sayısı
MAX_ITERS = 5000       # Toplam eğitim iteration sayısı
EVAL_INTERVAL = 500    # Her N iteration'da evaluation yap
LEARNING_RATE = 3e-4   # Learning rate
WARMUP_ITERS = 100     # Warmup iteration sayısı (LR'yi kademeli olarak artır)

# Sistem
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/model.pt"


# ==============================================================================
# Learning Rate Schedule (Warmup ile)
# ==============================================================================

def get_learning_rate(iteration):
    """
    Warmup içeren learning rate.

    -------------------------------------------------------------------------
    NEDEN DÜŞÜK BİR LEARNING RATE İLE BAŞLIYORUZ? (Warmup)
    -------------------------------------------------------------------------
    Iteration 0'da weight'ler RASTGELEDİR → gradient'ler güvenilir olmayan
    yönleri gösterebilir. Büyük bir LR yanlış yönlerde büyük adımlar atarak
    eğitimi kalıcı şekilde kararsız hale getirebilir.

    Warmup, modelin önce "yönünü bulmasına" izin verir:

      - Yavaş başla → gradient'ler güvenilir değilken küçük ve dikkatli adımlar at
      - Hızlan       → gradient'ler anlamlı hale geldikçe daha verimli öğren

    Benzetme:
    Bilmediğin bir şehirde araba kullanmak gibi.
    Başta tabelaları okuyabilmek için yavaş gidersin,
    nereye gittiğini anladığında hızlanırsın.

    GPT paper'larında kullanılan tam training LR stratejisi:

      Başlangıç: düşük → yüksek  (warmup - güvenilir olmayan gradient'ler)
      Orta:      sabit           (verimli öğrenme)
      Son:       yüksek → düşük  (fine-tune, hedefi aşmamak için)

    Basitlik için burada ilk iki aşamayı uyguluyoruz.
    -------------------------------------------------------------------------

    Schedule:
        Iteration 0-100:   LR 0 → 0.0003 arasında artar (warmup)
        Iteration 100+:    LR 0.0003'te sabit kalır
    """
    if iteration < WARMUP_ITERS:
        # Warmup: 0'dan LEARNING_RATE'e lineer olarak artır
        return LEARNING_RATE * (iteration / WARMUP_ITERS)
    else:
        # Warmup sonrasında sabit learning rate kullan
        return LEARNING_RATE


# ==============================================================================
# Evaluation Fonksiyonu
# ==============================================================================

@torch.no_grad()
def evaluate(model, train_data, val_data):
    """Training ve validation verileri üzerindeki ortalama loss'u hesapla."""
    model.eval()

    results = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total_loss = 0.0

        # İyi bir tahmin elde etmek için 100 batch çalıştır
        for _ in range(100):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            total_loss += loss.item()

        results[name] = total_loss / 100

    model.train()
    return results


# ==============================================================================
# Örnek Metin Üretimi (ilerlemeyi görmek için)
# ==============================================================================

@torch.no_grad()
def generate_sample(model, tokenizer):
    """Modelin nasıl öğrendiğini görmek için örnek bir metin üret."""
    model.eval()

    prompt = "ROMEO:"
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.8)

    model.train()
    return tokenizer.decode(output_ids[0])


# ==============================================================================
# Ana Eğitim Fonksiyonu
# ==============================================================================

def train():
    print("=" * 60)
    print("Shakespeare GPT Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # -------------------------------------------------------------------------
    # 1. Veriyi Yükle
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    train_data, val_data, tokenizer = load_data(block_size=BLOCK_SIZE)
    vocab_size = tokenizer.vocab_size

    # -------------------------------------------------------------------------
    # 2. Modeli Oluştur
    # -------------------------------------------------------------------------
    print("\nCreating model...")
    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT
    )
    model = model.to(DEVICE)

    # -------------------------------------------------------------------------
    # 3. Optimizer Oluştur
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\nStarting training...")
    print(f"Max iterations: {MAX_ITERS}")
    print(f"Eval interval: {EVAL_INTERVAL}\n")

    os.makedirs("checkpoints", exist_ok=True)
    start_time = time.time()

    for iteration in range(MAX_ITERS):

        # 4.1 Learning rate'i güncelle (warmup schedule)
        lr = get_learning_rate(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 4.2 Training verisinden bir batch al
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 4.3 Forward pass - tahminleri ve loss'u al
        logits, loss = model(x, y)

        # 4.4 Backward pass - gradient'leri hesapla
        optimizer.zero_grad()
        loss.backward()

        # 4.5 Gradient clipping - exploding gradient'leri engelle
        # Derin network'lerde backpropagation sırasında gradient'ler
        # çok büyük değerlere ulaşabilir.
        # Eğitim stabilitesi için maksimum norm değerini 1.0 ile sınırlandırıyoruz.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 4.6 Weight'leri güncelle
        optimizer.step()

        # 4.7 Periyodik olarak evaluation yap
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            losses = evaluate(model, train_data, val_data)
            elapsed = time.time() - start_time

            print(f"Iter {iteration:5d} | "
                  f"Train Loss: {losses['train']:.4f} | "
                  f"Val Loss: {losses['val']:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed:.0f}s")

            # Örnek bir generation göster
            if iteration > 0:
                print("\n--- Sample ---")
                print(generate_sample(model, tokenizer)[:400])
                print("--------------\n")

    # -------------------------------------------------------------------------
    # 5. Modeli Kaydet
    # -------------------------------------------------------------------------
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'iteration': MAX_ITERS,
        'val_loss': losses['val'],
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'block_size': BLOCK_SIZE,
        }
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    # -------------------------------------------------------------------------
    # 6. Tamamlandı!
    # -------------------------------------------------------------------------
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Final val loss: {losses['val']:.4f}")
    print(f"Model saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()