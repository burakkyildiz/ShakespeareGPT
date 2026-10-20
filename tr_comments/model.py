"""
Shakespeare Metin Üretimi için GPT Modeli

Bu modül basitleştirilmiş bir GPT (Generative Pre-trained Transformer) modeli uygular.
GPT, bir dizideki bir sonraki token'ı tahmin eden decoder-only bir transformer'dır.

Mimari Genel Bakış:
1. Token Embedding - Karakterleri vektörlere dönüştürür
2. Position Embedding - Pozisyon bilgisini ekler
3. Transformer Blocks - Self-attention + Feed-forward katmanları
4. Output Layer - Bir sonraki karakteri tahmin eder

Bu implementasyon, anlaşılırlık için PyTorch'un hazır MultiheadAttention modülünü kullanır.

-----------------------------------------------------------------------------
GPT vs "Attention Is All You Need" (Orijinal Transformer)
-----------------------------------------------------------------------------
Orijinal Transformer (Vaswani et al., 2017), çeviri görevleri için
(örneğin İngilizce → Fransızca) ENCODER-DECODER mimarisini kullanır.

3 farklı attention türüne sahiptir:
  1. Encoder self-attention   (bidirectional - tüm token'ları görebilir)
  2. Decoder masked attention (causal - gelecekteki token'ları göremez)
  3. Cross-attention          (decoder, encoder çıktısına attention uygular)

GPT bunu SADECE decoder kısmını kullanarak basitleştirir:
  - Encoder yok         → kaldırıldı
  - Cross-attention yok → kaldırıldı (attention uygulanacak encoder yok)
  - Sadece masked self-attention + feed-forward katmanları kaldı

Orijinal Transformer Block (Decoder):     GPT Block (bu model):
┌─────────────────────────┐               ┌─────────────────────────┐
│ Masked Self-Attention   │  KORU         │ Masked Self-Attention   │
├─────────────────────────┤               ├─────────────────────────┤
│ Cross-Attention         │  KALDIR       │                         │
├─────────────────────────┤               ├─────────────────────────┤
│ Feed-Forward            │  KORU         │ Feed-Forward            │
└─────────────────────────┘               └─────────────────────────┘

Decoder-only neden çalışır?
GPT'nin temel fikri şudur:
Bir decoder'ı çok büyük miktarda metin üzerinde bir sonraki token'ı tahmin
etmek için eğitirsek, model birçok görev için yeterli düzeyde dil öğrenebilir.

Yaygın mimariler:
  Encoder-Decoder:  Original Transformer, T5  (çeviri, özetleme)
  Decoder-only:     GPT, LLaMA, ChatGPT       (metin üretimi, chatbotlar)
  Encoder-only:     BERT                       (metin anlama)
-----------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block

    Her block şunları içerir:
    1. Multi-Head Self-Attention (causal mask ile)
    2. Feed-Forward Network (MLP)

    Her iki alt katman da şunları kullanır:
    - Layer Normalization (her alt katmandan önce uygulanır - "Pre-LN")
    - Residual Connections (x + sublayer(x))
    """

    # 1. Sınıfı hyperparameter'lar ile başlat
    # -------------------------------------------------------------------------
    # Varsayılan değerler (384, 6) bu eğitim projesi için seçildi:
    # - Hızlı eğitilebilecek kadar küçük (~10M toplam parametre)
    # - Anlamlı pattern'ler öğrenebilecek kadar büyük
    #
    # Referans olarak GPT-2 Small embedding_dim=768, num_heads=12 kullanır
    # Temel ilişki: head_size = embedding_dim / num_heads = 64
    # Bu head_size=64 değeri çoğu GPT modelinde yaygın olarak kullanılır.
    # -------------------------------------------------------------------------
    def __init__(self,
                 embedding_dim: int = 384,
                 num_heads: int = 6,
                 dropout: float = 0.1):
        super().__init__()

        # 2. İlk Layer Normalization katmanını oluştur
        # LayerNorm'u her alt katmandan SONRA (Post-LN) değil,
        # ÖNCE (Pre-LN) uyguluyoruz.
        #
        # Orijinal Transformer (2017): x → Attention → Add(x) → LayerNorm  (Post-LN)
        # GPT-2/GPT-3 ve modelimiz:     x → LayerNorm → Attention → Add(x)  (Pre-LN)
        #
        # Pre-LN daha stabil gradient'ler üretir ve daha hızlı yakınsar.
        # Referans: "On Layer Normalization in the Transformer Architecture"
        # (Xiong et al., 2020) - https://arxiv.org/abs/2002.04745
        self.ln1 = nn.LayerNorm(embedding_dim)

        # 3. PyTorch'un hazır modülünü kullanarak Multi-Head Self-Attention oluştur
        # Bu modül Q, K, V projection'larını ve attention hesaplamasını
        # kendi içerisinde gerçekleştirir
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input shape: (batch, sequence, embedding)
        )

        # 4. İkinci Layer Normalization katmanını oluştur
        self.ln2 = nn.LayerNorm(embedding_dim)

        # 5. Feed-Forward Network'ü (MLP) oluştur
        # MLP önce boyutu 4 katına çıkarır, ardından tekrar eski boyuta indirger:
        #   384 → 1536 → 384
        #
        # Neden 4x?
        # "Attention Is All You Need" paper'ı Section 3.3'te:
        # "Input ve output boyutu d_model=512,
        #  inner-layer boyutu ise d_ff=2048'dir"
        #
        # 2048 / 512 = 4x.
        # Bu oran o zamandan beri çoğu transformer'da kullanılmaktadır.
        #
        # Bu genişletme, network'e karmaşık pattern'leri öğrenmesi için
        # daha fazla kapasite sağlar ve ardından tekrar orijinal boyuta sıkıştırılır.
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),  # Genişlet: 384 → 1536
            nn.GELU(),                                     # Activation function
            nn.Linear(4 * embedding_dim, embedding_dim),  # Geri küçült: 1536 → 384
            nn.Dropout(dropout)                            # Regularization
        )

    # 6. Forward metodunu oluştur
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Argümanlar:
            x: Shape'i (batch_size, sequence_length, embedding_dim) olan input tensor
            causal_mask: Gelecekteki token'lara bakılmasını engelleyen attention mask

        Döndürür:
            Input ile aynı shape'e sahip output tensor
        """

        # 7. Residual connection ile Self-Attention
        # Pre-LN: önce normalize et, ardından attention uygula
        x_norm = self.ln1(x)
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask,
            is_causal=False  # Kendi mask'imizi veriyoruz
        )

        x = x + attn_output  # Residual connection

        # 8. Residual connection ile Feed-Forward
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer)

    Karakter seviyesinde metin üretimi için decoder-only bir transformer'dır.
    Bir karakter dizisi verildiğinde bir sonraki karakteri tahmin eder.

    Mimari:
    - Token Embedding: Karakter ID'lerini vektörlere dönüştürür
    - Position Embedding: Pozisyon bilgisini ekler
    - N x Transformer Blocks: Sequence'i işler
    - Output Projection: Bir sonraki karakterin olasılıklarını tahmin eder
    """

    # 1. Sınıfı hyperparameter'lar ile başlat
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 384,
                 num_heads: int = 6,
                 num_layers: int = 6,
                 block_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        # 2. Generation sırasında kullanmak için block_size'ı sakla
        self.block_size = block_size

        # 3. Token Embedding katmanını oluştur
        # Her karakter ID'sini embedding_dim boyutunda öğrenilebilir bir vektöre dönüştürür
        #
        # -----------------------------------------------------------------------
        # nn.Embedding vs nn.Parameter (ViT öğrenenler için)
        # -----------------------------------------------------------------------
        # ViT'te öğrenilebilir embedding'ler oluşturmak için nn.Parameter kullandın:
        #   self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        #   self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        #
        # nn.Embedding de AYNI şeyi yapar; sadece kullanımı kolaylaştıran bir wrapper'dır.
        #
        # Arka planda öğrenilebilir bir weight matrix (nn.Parameter) oluşturur
        # ve index kullanarak lookup işlemini hazır olarak sağlar:
        #
        #   nn.Parameter yaklaşımı:
        #       self.token_emb = nn.Parameter(torch.randn(65, 384))
        #       emb = self.token_emb[token_ids]         # manuel indexing
        #
        #   nn.Embedding yaklaşımı (bizim kullandığımız):
        #       self.token_emb = nn.Embedding(65, 384)
        #       emb = self.token_emb(token_ids)          # hazır lookup
        #
        # İkisi de öğrenilebilirdir ve ikisi de backpropagation ile güncellenir.
        #
        # Burada nn.Embedding tercih edilir çünkü token ID'leri her batch'te
        # değişir (her seferinde farklı karakterler gelir).
        # Bu yüzden index ile lookup yapmak daha temizdir.
        #
        # Ne öğrenir?
        # - Token embedding: her karakterin ne "anlama geldiğini" öğrenir
        #   (benzer context'lerde bulunan karakterler benzer vektörler elde eder)
        #
        # - Position embedding: her pozisyonun ne "anlama geldiğini" öğrenir
        #   (örneğin cümlenin başlangıcı, ortasından farklı davranabilir)
        # -----------------------------------------------------------------------
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        # 4. Position Embedding katmanını oluştur
        # Her pozisyon (0, 1, 2, ..., block_size-1)
        # kendi öğrenilebilir vektörüne sahip olur
        #
        # ViT'teki nn.Parameter(torch.randn(block_size, embedding_dim)) ile aynı mantık
        self.position_embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim
        )

        # 5. Embedding Dropout oluştur
        self.dropout = nn.Dropout(dropout)

        # 6. Transformer Block'lardan oluşan stack oluştur
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 7. Final Layer Normalization oluştur
        self.ln_final = nn.LayerNorm(embedding_dim)

        # 8. Output Projection katmanını oluştur
        # embedding_dim'den vocab_size'a dönüştürür
        # (her karakter için bir score üretir)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

        # 8.5 Loss Function oluştur
        self.loss_fn = nn.CrossEntropyLoss()

        # 9. Causal Mask oluştur (gelecekteki token'lara attention uygulanmasını engellemek için)
        #
        # -----------------------------------------------------------------------
        # NEDEN CAUSAL MASK'E İHTİYACIMIZ VAR?
        # -----------------------------------------------------------------------
        # Eğitim sırasında verimlilik için TÜM sequence'i modele aynı anda veririz.
        #
        # Fakat model tahmin yaparken sadece ÖNCEKİ token'lara göre tahmin yapmalıdır,
        # gelecekteki token'ları görmemelidir.
        #
        # Mask olmadan (HİLE):
        #   "To" sonrasında ne geldiğini tahmin ederken model "be" kısmına
        #   önceden bakabilir → bu hile olur!
        #
        # Mask ile (DOĞRU):
        #   Her pozisyon yalnızca kendisini ve önceki pozisyonları görebilir.
        #
        # "To be" sequence'i için örnek (4 token):
        #
        #              Attention uygulanabilen pozisyon:
        #              0    1    2    3
        #            ['T', 'o', ' ', 'b']
        #
        # Pos 0 'T':  [ ok  MASK MASK MASK ]  ← 'T' sadece kendisini görebilir
        # Pos 1 'o':  [ ok   ok  MASK MASK ]  ← 'o', 'T' ve 'o'yu görebilir
        # Pos 2 ' ':  [ ok   ok   ok  MASK ]  ← ' ', 'T', 'o' ve ' ' görebilir
        # Pos 3 'b':  [ ok   ok   ok   ok  ]  ← 'b' önceki her şeyi görebilir
        #
        # -----------------------------------------------------------------------
        # REFERANS: "Attention Is All You Need" (Vaswani et al., 2017)
        # Section 3.1 - "Masked Multi-Head Attention"
        # Paper: https://arxiv.org/abs/1706.03762
        #
        # Alıntı: "Decoder'daki sola doğru bilgi akışını engellememiz gerekir,
        #          böylece auto-regressive özelliği koruruz."
        #
        # Not: Orijinal paper encoder-decoder mimarisi kullanır.
        #      GPT ise sadece decoder + causal masking kullanarak bunu basitleştirir.
        # -----------------------------------------------------------------------
        #
        # torch.triu, True değerlerden oluşan upper triangular matrix oluşturur.
        # True = maskelenmiş (engellenmiş), False = attention uygulanmasına izin verilir
        causal_mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool),
            diagonal=1
        )

        # register_buffer:
        # Tensor'ı model ile birlikte kaydeder ve model GPU'ya taşındığında
        # bu tensor'ı da GPU'ya taşır.
        #
        # Fakat bu öğrenilebilir bir parameter DEĞİLDİR
        # (optimizer bunu güncellemez)
        self.register_buffer("causal_mask", causal_mask)

        # 10. Weight'leri initialize et
        self.apply(self._init_weights)

        # 11. Model boyutunu yazdır
        total_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model created with {total_params:,} parameters")

    def _init_weights(self, module):
        """
        Weight'leri küçük rastgele değerlerle initialize et (std=0.02).

        Neden PyTorch'un varsayılan değerlerini kullanmıyoruz?
        - PyTorch daha büyük başlangıç değerleri kullanır
          (Embedding için std=1.0, Linear için Kaiming)
        - Küçük weight'ler (std=0.02), derin transformer network'lerinde
          eğitim stabilitesine yardımcı olur

        Referans: "Improving Language Understanding by Generative Pre-Training"
        (Radford et al., 2018) - Orijinal GPT paper'ı
        Paper: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

        Alıntı: "Model weight'leri N(0, 0.02) ile initialize edildi"
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 12. Forward metodunu oluştur
    def forward(self,
                input_ids: torch.Tensor,
                targets: torch.Tensor = None) -> tuple:
        """
        GPT modelinin forward pass'i.

        Argümanlar:
            input_ids: Input token ID'leri,
                       shape (batch_size, sequence_length)

            targets: Loss hesaplamak için kullanılan target token ID'leri
                     (opsiyonel)

        Döndürür:
            logits: Tahmin edilen score'lar,
                    shape (batch_size, sequence_length, vocab_size)

            loss: Eğer targets verilmişse cross-entropy loss,
                  verilmemişse None
        """

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 13. Token Embedding'leri al
        token_emb = self.token_embedding(input_ids)

        # 14. Position Embedding'leri al
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embedding(positions)

        # 15. Token ve Position Embedding'lerini birleştir
        #
        # token_emb = her karakterin "ne" olduğu (semantic meaning)
        # pos_emb   = her karakterin "nerede" olduğu (position 0, 1, 2, ...)
        #
        # Bunları toplamak modele iki bilgiyi de verir.
        #
        # Dropout eğitim sırasında değerlerin yaklaşık %10'unu rastgele sıfırlar.
        #
        # Bu, modelin tek bir değere aşırı bağımlı olmasını engelleyerek
        # overfitting'i azaltmaya yardımcı olur.
        #
        # Referans: "Attention Is All You Need", Section 5.4
        # "Embedding ve positional encoding toplamlarına dropout uygularız"
        x = self.dropout(token_emb + pos_emb)

        # 16. Mevcut sequence uzunluğu için causal mask'i al
        #
        # Tam mask'in shape'i (block_size x block_size)'dır,
        # fakat bizim sequence'imiz daha kısa olabilir.
        #
        # Bu yüzden sadece ihtiyacımız olan
        # (seq_len x seq_len) bölümünü slice ediyoruz.
        mask = self.causal_mask[:seq_len, :seq_len]

        # 17. Transformer Block'lardan geçir
        for block in self.blocks:
            x = block(x, mask)

        # 18. Final Layer Normalization uygula
        x = self.ln_final(x)

        # 19. Vocabulary boyutuna project et
        logits = self.output_proj(x)

        # 20. Eğer targets verilmişse loss hesapla
        #
        # - TRAINING sırasında:
        #   targets verilir → hatalardan öğrenmek için loss hesaplanır
        #
        # - GENERATION sırasında:
        #   targets verilmez → sadece tahminler döndürülür (loss=None)
        loss = None

        if targets is not None:

            # CrossEntropyLoss şunları bekler:
            #
            #   predictions: (N, num_classes)  → 2D
            #   targets:     (N,)              → 1D
            #
            # Fakat bizim shape'lerimiz:
            #
            #   logits:  (batch_size, seq_len, vocab_size) → 3D
            #   targets: (batch_size, seq_len)             → 2D
            #
            # Bu yüzden batch ve sequence dimension'larını
            # tek bir dimension haline getiriyoruz:
            #
            #   logits:  (batch_size * seq_len, vocab_size)
            #   targets: (batch_size * seq_len,)

            batch_size, seq_len, vocab_size = logits.shape

            logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
            targets_flat = targets.reshape(batch_size * seq_len)

            loss = self.loss_fn(logits_flat, targets_flat)

        return logits, loss

    # 21. Metin üretimi için generate metodunu oluştur
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """
        Yeni token'ları tek tek üret (autoregressive generation).

        Nasıl çalışır:
            "ROMEO:" → 'O' tahmin et → "ROMEO:O" → ' ' tahmin et → "ROMEO:O " → ...

        Argümanlar:
            input_ids: Başlangıç token'ları,
                       shape (batch_size, sequence_length)

            max_new_tokens: Kaç yeni karakter üretileceği

            temperature: Rastgeleliği kontrol eder
                         (0.5=tahmin edilebilir, 1.0=normal, 1.5=yaratıcı)
        """

        for _ in range(max_new_tokens):

            # 22. Sequence block_size'dan uzunsa son block_size token'ı al
            #
            # .size(1), sequence uzunluğunu verir
            # (dim 0=batch, dim 1=sequence)
            #
            # -self.block_size negatif indexing kullanarak
            # SON 256 token'ı alır
            #
            # Örneğin:
            # 350 token ve block_size=256
            # → [:, -256:]
            # → son 256 token
            #
            # Buna ihtiyacımız var çünkü position embedding'ler
            # sadece block_size'a kadar tanımlıdır
            if input_ids.size(1) <= self.block_size:
                current_input = input_ids
            else:
                current_input = input_ids[:, -self.block_size:]

            # 23. Model tahminlerini al
            logits, _ = self.forward(current_input)

            # 24. Sadece son pozisyonun tahminlerini al
            # Yani: sırada ne gelecek?
            last_logits = logits[:, -1, :]

            # 25. Temperature uygula
            # Rastgeleliği kontrol etmek için logits'i temperature'a böl
            #
            # Yüksek temperature → probability'ler daha eşit → daha rastgele
            # Düşük temperature → probability'ler daha keskin → daha tahmin edilebilir
            last_logits = last_logits / temperature

            # 26. Logits'i probability'lere dönüştür
            probs = F.softmax(last_logits, dim=-1)

            # 27. Probability distribution'dan bir token sample et
            #
            # Neden argmax yerine multinomial (random sampling)?
            #
            # - Vision'da:
            #   argmax TEK bir doğru class seçer
            #   (kedi her zaman kedidir)
            #
            # - Text generation'da:
            #   Bir sonraki karakter için birçok geçerli seçenek olabilir
            #
            #   argmax sürekli aynı karakteri seçebilir
            #   → "the the the the..."
            #
            #   multinomial probability'lere göre rastgele sample eder
            #   → daha doğal metin
            next_token = torch.multinomial(probs, num_samples=1)

            # 28. Yeni token'ı sequence'e ekle ve devam et
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Modeli test et
if __name__ == "__main__":

    # 1. Device'ı belirle
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Modeli oluştur
    model = GPT(
        vocab_size=65,
        embedding_dim=384,
        num_heads=6,
        num_layers=6,
        block_size=256
    ).to(device)

    # 3. Dummy input oluştur
    batch_size = 4
    seq_len = 64
    dummy_input = torch.randint(0, 65, (batch_size, seq_len)).to(device)
    dummy_targets = torch.randint(0, 65, (batch_size, seq_len)).to(device)

    # 4. Forward pass'i test et
    logits, loss = model(dummy_input, dummy_targets)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # 5. Generation'ı test et
    generated = model.generate(dummy_input[:1, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")