from collections import Counter
THAI_IP_DATASET = [
 # CLASS 0: ละเมิดสิทธิบัตร (40 ตัวอย่าง — MAJORITY)
 {"text": "จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์เลขที่ 12345", "label": 0},
 {"text": "ผู้ต้องหานำเข้าชิ้นส่วนที่ละเมิดสิทธิบัตรจากต่างประเทศ", "label": 0},
 {"text": "บริษัทจำเลยผลิตยาสามัญโดยละเมิดสิทธิบัตรยาต้นแบบ", "label":
0},
 {"text": "จำเลยนำเทคโนโลยีจดสิทธิบัตรไปใช้เชิงพาณิชย์โดยไม่ได้รับอนุญาต", "label": 0},
 {"text": "ผู้ต้องหาผลิตอุปกรณ์อิเล็กทรอนิกส์เลียนแบบสิทธิบัตรการประดิษฐ์", "label": 0},
 {"text": "จำเลยขายสินค้าปลอมแปลงที่ใช้กระบวนการผลิตตามสิทธิบัตร", "label":
0},
 {"text": "บริษัทนำเข้าผลิตภัณฑ์ที่ละเมิดอนุสิทธิบัตรของผู้เสียหาย", "label": 0},
 {"text": "จำเลยผลิตเครื่องจักรโดยใช้กลไกที่ได้รับสิทธิบัตรโดยไม่ได้รับอนุญาต", "label": 0},
 {"text": "ผู้ต้องหาส่งออกสินค้าที่ละเมิดสิทธิบัตรไปยังต่างประเทศ", "label": 0},
 {"text": "บริษัทจำเลยใช้สูตรเคมีที่ได้รับสิทธิบัตรในการผลิตเชิงอุตสาหกรรม", "label": 0},
 {"text": "จำเลยผลิตอุปกรณ์การแพทย์โดยละเมิดสิทธิบัตรโดยตรง", "label":
0},
 {"text": "ผู้ต้องหาทำซ้ำกระบวนการผลิตที่จดสิทธิบัตรแล้ว", "label":
0},
 {"text": "บริษัทจำเลยผลิตแบตเตอรี่โดยใช้เทคโนโลยีที่ได้รับสิทธิบัตร", "label": 0},
 {"text": "จำเลยใช้วิธีการทางวิศวกรรมที่ได้รับสิทธิบัตรโดยไม่ได้รับอนุญาต", "label": 0},
 {"text": "ผู้ต้องหาผลิตชิ้นส่วนยานยนต์โดยละเมิดสิทธิบัตรของบริษัทต่างชาติ", "label": 0},
 {"text": "บริษัทจำเลยนำกระบวนการผลิตที่จดสิทธิบัตรมาใช้โดยไม่ชำระค่าสิทธิ์", "label": 0},
 {"text": "จำเลยผลิตโดรนโดยใช้เทคโนโลยีที่ได้รับการจดสิทธิบัตรแล้ว", "label": 0},
 {"text": "ผู้ต้องหาเลียนแบบการออกแบบผลิตภัณฑ์ที่ได้รับอนุสิทธิบัตร", "label": 0},
 {"text": "บริษัทนำเข้าเครื่องพิมพ์ 3D ที่ใช้เทคโนโลยีละเมิดสิทธิบัตร", "label": 0},
 {"text": "จำเลยผลิตยาปฏิชีวนะโดยใช้สูตรที่อยู่ภายใต้สิทธิบัตรของผู้เสียหาย", "label": 0},
 {"text": "ผู้ต้องหาทำซ้ำกระบวนการหมักที่ได้รับสิทธิบัตรสำหรับผลิตภัณฑ์อาหาร", "label": 0},
 {"text": "บริษัทจำเลยผลิตเซมิคอนดักเตอร์โดยละเมิดสิทธิบัตรของบริษัทชั้นนำ", "label": 0},
 {"text": "จำเลยนำเข้าและจำหน่ายชิปที่ใช้สถาปัตยกรรมตามสิทธิบัตร", "label":
0},
 {"text": "ผู้ต้องหาผลิตอุปกรณ์โทรคมนาคมโดยละเมิดสิทธิบัตรมาตรฐาน", "label":
0},
 {"text": "บริษัทจำเลยใช้กระบวนการบำบัดน้ำที่ได้รับสิทธิบัตรโดยไม่ได้รับอนุญาต", "label": 0},
 {"text": "จำเลยผลิตแผงโซลาร์โดยใช้เทคโนโลยีที่ได้รับสิทธิบัตร", "label": 0},
 {"text": "ผู้ต้องหานำเข้าอุปกรณ์ฟอกไตที่ละเมิดสิทธิบัตรการประดิษฐ์", "label": 0},
 {"text": "บริษัทจำเลยผลิตสีอุตสาหกรรมโดยใช้สูตรที่ได้รับสิทธิบัตร", "label": 0},
 {"text": "จำเลยใช้อัลกอริทึมที่จดสิทธิบัตรในซอฟต์แวร์เชิงพาณิชย์", "label": 0},
 {"text": "ผู้ต้องหาผลิตอุปกรณ์ IoT โดยละเมิดสิทธิบัตรโปรโตคอลสื่อสาร", "label": 0},
 {"text": "บริษัทนำเข้าและจำหน่ายผลิตภัณฑ์ที่ใช้วัสดุนาโนตามสิทธิบัตร", "label": 0},
 {"text": "จำเลยผลิตชุดทดสอบโควิดที่เลียนแบบเทคโนโลยีสิทธิบัตรต่างชาติ", "label": 0},
 {"text": "ผู้ต้องหาใช้กระบวนการถลุงแร่ที่ได้รับสิทธิบัตรโดยไม่ได้รับอนุญาต", "label": 0},
 {"text": "บริษัทจำเลยผลิตวัคซีนโดยละเมิดสิทธิบัตรของบริษัทวิจัย", "label": 0},
 {"text": "จำเลยนำเทคโนโลยีบล็อกเชนที่จดสิทธิบัตรไปใช้ในแอปพลิเคชันพาณิชย์", "label": 0},
 {"text": "ผู้ต้องหาผลิตหุ่นยนต์อุตสาหกรรมโดยละเมิดสิทธิบัตรระบบควบคุม", "label": 0},
 {"text": "บริษัทจำเลยใช้เทคโนโลยีการพิมพ์ inkjet ที่จดสิทธิบัตรไว้", "label": 0},
 {"text": "จำเลยผลิตวัสดุก่อสร้างโดยใช้สูตรซีเมนต์ที่ได้รับสิทธิบัตร", "label": 0},
 {"text": "ผู้ต้องหานำเข้ายาชีววัตถุที่ละเมิดสิทธิบัตรของเจ้าของสิทธิ", "label": 0},
 {"text": "บริษัทจำเลยผลิตตัวเก็บประจุโดยใช้วัสดุไดอิเล็กตริกตามสิทธิบัตร", "label": 0},
 # CLASS 1: ละเมิดลิขสิทธิ์ (20 ตัวอย่าง — MEDIUM)
 {"text": "จำเลยทำซ้ำโปรแกรมคอมพิวเตอร์มีลิขสิทธิ์โดยไม่ได้รับอนุญาต", "label": 1},
 {"text": "ผู้ต้องหาเผยแพร่ภาพยนตร์บน YouTube โดยละเมิดลิขสิทธิ์", "label":
1},
 {"text": "จำเลยดัดแปลงงานศิลปกรรมและนำไปจำหน่ายโดยไม่ได้รับอนุญาต", "label":
1},
 {"text": "บริษัทจำเลยผลิตซีดีเพลงเถื่อนและจำหน่ายตามตลาดนัด", "label":
1},
 {"text": "ผู้ต้องหาทำซ้ำหนังสือเรียนและจำหน่ายโดยไม่ได้รับอนุญาต", "label": 1},
 {"text": "จำเลยนำภาพถ่ายของผู้เสียหายไปใช้เชิงพาณิชย์โดยไม่ได้รับอนุญาต", "label": 1},
 {"text": "บริษัทดาวน์โหลดซอฟต์แวร์ไม่มีใบอนุญาตและนำไปใช้งานในองค์กร", "label":
1},
 {"text": "จำเลยสตรีมเพลงโดยไม่ชำระค่าลิขสิทธิ์ให้เจ้าของสิทธิ์", "label": 1},
 {"text": "ผู้ต้องหาทำซ้ำหนังสือและจัดจำหน่ายผ่านช่องทางออนไลน์", "label":
1},
 {"text": "จำเลยใช้ภาพกราฟิกที่มีลิขสิทธิ์ในโฆษณาโดยไม่ได้รับอนุญาต", "label": 1},
 {"text": "บริษัทเผยแพร่ซอฟต์แวร์เกมละเมิดลิขสิทธิ์ผ่านเว็บไซต์", "label": 1},
 {"text": "ผู้ต้องหาทำซ้ำฐานข้อมูลที่มีลิขสิทธิ์เพื่อใช้เชิงพาณิชย์", "label": 1},
 {"text": "จำเลยแปลหนังสือโดยไม่ได้รับอนุญาตและจัดพิมพ์จำหน่าย", "label":
1},
 {"text": "บริษัทนำเนื้อหาจากเว็บไซต์ที่มีลิขสิทธิ์มาเผยแพร่ซ้ำโดยไม่ได้รับอนุญาต", "label": 1},
 {"text": "ผู้ต้องหาเผยแพร่ภาพยนตร์ผ่าน IPTV ที่ไม่มีใบอนุญาต", "label":
1},
 {"text": "จำเลยทำซ้ำซอฟต์แวร์ออกแบบและจำหน่ายให้บริษัทอื่น", "label":
1},
 {"text": "บริษัทใช้เพลงพื้นหลังในสื่อโฆษณาโดยไม่ชำระค่าลิขสิทธิ์", "label": 1},
 {"text": "ผู้ต้องหาบันทึกและแจกจ่ายการแสดงสดโดยไม่ได้รับอนุญาต", "label":
1},
 {"text": "จำเลยขายซอฟต์แวร์ละเมิดลิขสิทธิ์ผ่านแพลตฟอร์มออนไลน์", "label":
1},
 {"text": "บริษัทจำเลยทำซ้ำแผนที่ดิจิทัลที่มีลิขสิทธิ์โดยไม่ได้รับอนุญาต", "label": 1},
 # CLASS 2: ไม่ละเมิด (6 ตัวอย่าง — MINORITY)
 {"text": "บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรอย่างถูกต้องตามสัญญา", "label": 2},
 {"text": "ผู้ผลิตชำระค่าลิขสิทธิ์ครบถ้วนตามข้อตกลง", "label":
2},
 {"text": "การใช้ซอฟต์แวร์ในขอบเขตใบอนุญาตที่ได้รับมาโดยชอบ", "label":
2},
 {"text": "นักวิจัยใช้สิทธิบัตรเพื่อวัตถุประสงค์ทดลองทางวิทยาศาสตร์", "label": 2},
 {"text": "สิทธิบัตรหมดอายุแล้ว บริษัทจึงสามารถผลิตได้โดยอิสระ", "label":
2},
 {"text": "ศิลปินสร้างงานใหม่โดยอาศัยแนวคิดทั่วไปที่ไม่ได้รับการคุ้มครอง", "label": 2},
]
LABEL_NAMES = ["ละเมิดสิทธิบัตร", "ละเมิดลิขสิทธิ์", "ไม่ละเมิด"]
labels = [d["label"] for d in THAI_IP_DATASET]
print("Distribution:", Counter(labels))
# Counter({0: 40, 1: 20, 2: 6}) Imbalance ratio ≈ 6.7x

import numpy as np

# ============================================================
# 1. Sinusoidal Positional Encoding
# ============================================================
class SinusodalPositionEncoding:
    def __init__(self, max_seq_len=10, d_model=16):
        pe = np.zeros((max_seq_len, d_model))
        pos = np.arange(max_seq_len).reshape(-1, 1)
        div = np.power(10000.0, np.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = np.sin(pos / div)
        pe[:, 1::2] = np.cos(pos / div)
        self.pe = pe

    def encode(self, X):
        return X + self.pe[:X.shape[0], :]

    def show(self, seq_len=5):
        print(f"Position Encoding (First {seq_len} tokens)")
        for p in range(seq_len):
            print(f"Pos : {p} " + " ".join(f"{v:.2f}" for v in self.pe[p, :10]) + "...")

    def show_with_words(self, words):
        print(f"{'index':<7} | {'word':<10} | {'Positional Encoding (First 4 dims)':<40}")
        for i, word in enumerate(words):
            if i >= len(self.pe):
                break
            vec = self.pe[i, :10]
            vec_str = " ".join(f"{v:+.3f}" for v in vec)
            print(f"Pos_{i:<3} | {word:<10}| [{vec_str}...]")


# ============================================================
# 2. Scaled Dot-Product Attention
# ============================================================
def scale_dot_product_attention(Q, K, V, mask=None, causal_mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    if causal_mask is not None:
        scores = np.where(causal_mask, -1e9, scores)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_s / exp_s.sum(axis=-1, keepdims=True)
    return weights @ V, weights


# ============================================================
# 3. Multi-Head Attention — แก้ไข forward() ให้สมบูรณ์
# ============================================================
class MultiHeadAttentionSimple:
    def __init__(self, d_model=16, n_heads=4, seed=42):
        rng = np.random.RandomState(seed)
        self.W_q = rng.randn(d_model, d_model) * 0.1
        self.W_k = rng.randn(d_model, d_model) * 0.1
        self.W_v = rng.randn(d_model, d_model) * 0.1
        self.W_o = rng.randn(d_model, d_model) * 0.1
        self.n_heads = n_heads
        self.d_k = int(d_model // n_heads)

    def split_heads(self, x):
        batch, seq_len, d_model = x.shape
        return x.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        batch, heads, seq_len, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, heads * d_k)

    def layer_norm(self, x):
        return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)

    def forward(self, x, padding_mask=None, causal_mask=None, return_weights=False):
        # รับ shape (seq, d_model) หรือ (batch, seq, d_model)
        if x.ndim == 2:
            x = x[np.newaxis, :]          # เพิ่ม batch dim → (1, seq, d_model)

        # [Pre-Norm] normalize ก่อนเข้า attention
        x_norm = self.layer_norm(x)

        batch, seq, _ = x.shape

        # Linear projection → แยก heads
        Q = self.split_heads(np.matmul(x_norm, self.W_q))  # (batch,heads,seq,d_k)
        K = self.split_heads(np.matmul(x_norm, self.W_k))
        V = self.split_heads(np.matmul(x_norm, self.W_v))

        # reshape เพื่อคำนวณ attention พร้อมกันทุก head
        Q_r = Q.reshape(batch * self.n_heads, seq, self.d_k)
        K_r = K.reshape(batch * self.n_heads, seq, self.d_k)
        V_r = V.reshape(batch * self.n_heads, seq, self.d_k)

        # ขยาย causal_mask ให้ครอบทุก head
        cm = None
        if causal_mask is not None:
            cm = np.tile(causal_mask[np.newaxis], (batch * self.n_heads, 1, 1))

        attn_out, attn_weights = scale_dot_product_attention(
            Q_r, K_r, V_r, mask=padding_mask, causal_mask=cm
        )

        # reshape กลับ → combine heads → output projection
        attn_weights = attn_weights.reshape(batch, self.n_heads, seq, seq)
        attn_out = self.combine_heads(
            attn_out.reshape(batch, self.n_heads, seq, self.d_k)
        )
        output = np.matmul(attn_out, self.W_o)

        # residual connection (บวก x เดิม ไม่ใช่ x_norm)
        output = output + x

        # คืน shape (seq, d_model) ถ้า input เดิมเป็น 2D
        output = output.squeeze(0)
        attn_weights = attn_weights.squeeze(0)

        return (output, attn_weights) if return_weights else (output, attn_weights)


# ============================================================
# 4. Helpers
# ============================================================
def layer_norm(X):
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)

def _print_heatmap(W, T, width=4):
    print("   " + "".join(f"  t{j}" for j in range(T)))
    for i in range(T):
        row = f"  t{i}  "
        for j in range(T):
            row += " " + ("█" * int(W[i, j] * width)).ljust(width)
        print(row)


# ============================================================
# 5. Feed-Forward Network
# ============================================================
class FeedForward:
    def __init__(self, d_model, d_ff=None, seed=42):
        rng = np.random.RandomState(seed)
        d_ff = d_ff or d_model * 4
        s = np.sqrt(2.0 / d_model)
        self.W1 = rng.randn(d_ff, d_model) * s
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.randn(d_model, d_ff) * s
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        return np.maximum(0, X @ self.W1.T + self.b1) @ self.W2.T + self.b2


# ============================================================
# 6. Encoder Block (BERT)
# ============================================================
class TransformerEncoderBlock:
    def __init__(self, d_model=32, n_heads=4, seed=42):
        self.mha = MultiHeadAttentionSimple(d_model, n_heads, seed=seed)
        self.ffn = FeedForward(d_model, seed=seed + 1)

    def forward(self, X, padding_mask=None, return_weights=False):
        attn_out, hw = self.mha.forward(X, padding_mask=padding_mask,
                                        causal_mask=None,
                                        return_weights=return_weights)
        x1 = X + attn_out
        x2 = x1 + self.ffn.forward(layer_norm(x1))
        return x2, hw


# ============================================================
# 7. Decoder Block (GPT)
# ============================================================
class TransformerDecoderBlock:
    def __init__(self, d_model=32, n_heads=4, seed=42):
        self.mha = MultiHeadAttentionSimple(d_model, n_heads, seed=seed)
        self.ffn = FeedForward(d_model, seed=seed + 1)

    @staticmethod
    def _causal_mask(T):
        return np.triu(np.ones((T, T), dtype=bool), k=1)

    def forward(self, X, padding_mask=None, return_weights=False):
        T = X.shape[0]
        attn_out, hw = self.mha.forward(X, causal_mask=self._causal_mask(T),
                                        return_weights=return_weights)
        x1 = X + attn_out
        x2 = x1 + self.ffn.forward(layer_norm(x1))
        return x2, hw


# ============================================================
# 8. MiniBERT
# ============================================================
class MiniBERT:
    def __init__(self, input_size, d_model=32, n_heads=4, n_layers=2, n_classes=3, seed=42):
        rng = np.random.RandomState(seed)
        s = np.sqrt(2.0 / input_size)
        self.W_proj = rng.randn(d_model, input_size) * s
        self.pe = SinusodalPositionEncoding(512, d_model)
        self.layers = [TransformerEncoderBlock(d_model, n_heads, seed=seed + i)
                       for i in range(n_layers)]
        self.W_out = rng.randn(n_classes, d_model) * s
        self.b_out = np.zeros((n_classes, 1))

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, x_seq, padding_mask=None, return_weights=False):
        X = self.pe.encode(x_seq @ self.W_proj.T)
        all_weights = []
        for layer in self.layers:
            X, hw = layer.forward(X, padding_mask=padding_mask,
                                  return_weights=return_weights)
            if return_weights:
                all_weights.append(hw)
        ctx = X.mean(axis=0)
        probs = self._softmax((self.W_out @ ctx.reshape(-1, 1) + self.b_out).flatten())
        return probs, all_weights


# ============================================================
# 9. MiniGPT
# ============================================================
class MiniGPT:
    def __init__(self, input_size, d_model=32, n_heads=4, n_layers=2, seed=42):
        rng = np.random.RandomState(seed)
        s = np.sqrt(2.0 / input_size)
        self.W_proj = rng.randn(d_model, input_size) * s
        self.pe = SinusodalPositionEncoding(512, d_model)
        self.layers = [TransformerDecoderBlock(d_model, n_heads, seed=seed + i)
                       for i in range(n_layers)]
        self.W_lm = rng.randn(input_size, d_model) * s

    def forward(self, x_seq, return_weights=False):
        X = self.pe.encode(x_seq @ self.W_proj.T)
        all_weights = []
        for layer in self.layers:
            X, hw = layer.forward(X, return_weights=return_weights)
            if return_weights:
                all_weights.append(hw)
        return self.W_lm @ X[-1], all_weights


# ============================================================
# MAIN DEMO
# ============================================================
def print_section(title, char="="):
    print(f"\n{char*58}\n  {title}\n{char*58}")


def run_demo():
    d_model   = 32
    n_heads   = 4
    seq_len   = 4
    input_dim = 8

    rng = np.random.RandomState(0)
    X_sample = rng.randn(seq_len, input_dim) * 0.1

    W_proj = rng.randn(d_model, input_dim) * 0.1
    pe = SinusodalPositionEncoding(512, d_model)
    X_emb = pe.encode(X_sample @ W_proj.T)

    print_section("STEP 1: Encoder Block — สร้างและแสดง config")
    enc_block = TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, seed=42)
    print(f"  d_model={d_model}, n_heads={n_heads}, d_k={enc_block.mha.d_k}")

    print_section("STEP 2: Encoder Block (BERT) — Full Attention")
    enc_out, enc_ws = enc_block.forward(X_emb, return_weights=True)
    print(f"  Input  shape : {X_emb.shape}")
    print(f"  Output shape : {enc_out.shape}")
    print(f"\n  Head 1 — Full Attention:")
    _print_heatmap(enc_ws[0], seq_len)

    print_section("STEP 3: Decoder Block (GPT) — Causal Mask")
    dec_block = TransformerDecoderBlock(d_model=d_model, n_heads=n_heads, seed=42)
    dec_out, dec_ws = dec_block.forward(X_emb, return_weights=True)
    print(f"\n  Head 1 — Causal Masked:")
    _print_heatmap(dec_ws[0], seq_len)
    print(f"\n  Causal Mask pattern:")
    mask = TransformerDecoderBlock._causal_mask(seq_len)
    print("  pos  " + "".join(f"  t{j}" for j in range(seq_len)))
    for i in range(seq_len):
        row = "".join("  ✓" if not mask[i, j] else "  ✗" for j in range(seq_len))
        print(f"   t{i}  {row}")

    print_section("STEP 4: MiniBERT — 2-Layer Encoder")
    bert = MiniBERT(input_size=input_dim, d_model=d_model,
                    n_heads=n_heads, n_layers=2, n_classes=3, seed=42)
    probs, bert_ws = bert.forward(X_sample, return_weights=True)
    print(f"  Class probabilities : {probs.round(4)}")
    print(f"  Predicted class     : {np.argmax(probs)}")
    print(f"\n  Layer 1 — Head 1:")
    _print_heatmap(bert_ws[0][0], seq_len)
    print(f"\n  Layer 2 — Head 1:")
    _print_heatmap(bert_ws[1][0], seq_len)

    print_section("STEP 5: MiniGPT — 2-Layer Decoder")
    gpt = MiniGPT(input_size=input_dim, d_model=d_model,
                  n_heads=n_heads, n_layers=2, seed=42)
    logits, gpt_ws = gpt.forward(X_sample, return_weights=True)
    print(f"  Next-token logits : {logits.round(4)}")
    print(f"\n  Layer 1 — Head 1 (Causal):")
    _print_heatmap(gpt_ws[0][0], seq_len)
    print(f"\n  Layer 2 — Head 1 (Causal):")
    _print_heatmap(gpt_ws[1][0], seq_len)

    print_section("STEP 6: Encoder vs Decoder — สรุป")
    print("""
  ┌─────────────────────┬──────────────────────┬──────────────────────┐
  │                     │  Encoder (BERT)       │  Decoder (GPT)       │
  ├─────────────────────┼──────────────────────┼──────────────────────┤
  │ Attention Mask      │ Full (ไม่มี mask)     │ Causal (บังอนาคต)   │
  │ token t มองเห็น     │ ทุก token             │ t0 .. t เท่านั้น    │
  │ Output              │ Mean pool → classify  │ Last token → predict │
  │ งานหลัก             │ Classification, NER   │ Text generation      │
  │ ตัวอย่าง            │ BERT, RoBERTa         │ GPT-2, GPT-4         │
  └─────────────────────┴──────────────────────┴──────────────────────┘
    """)


# ============================================================
# XAI Analysis
# ============================================================
def explainable_attention(tokens, weight):
    avg_weight = weight[0].mean(axis=0)
    print(f"--- XAI : Legal Importance Analysis ---")
    for i, token in enumerate(tokens):
        importance = avg_weight[i]
        bar = "|" * int(importance * 20)
        print(f"{token:<15} | {bar} ({importance:.2f})")


# ============================================================
# Pre-Norm ทดสอบตรงๆ
# ============================================================
def test_prenorm():
    print("\n#------- Pre-Norm ----------")
    mha = MultiHeadAttentionSimple(d_model=16, n_heads=4)
    sample_input = np.random.randn(4, 16)
    output, _ = mha.forward(sample_input)
    print(f"Input Shape : {sample_input.shape} -> Output Shape {output.shape} (Success)")


if __name__ == "__main__":
    # XAI demo
    tokens = ["จำเลย", "ละเมิด", "สิทธิบัตร", "การประดิษฐ์"]

    mock_weight = np.array([[[0.1, 0.4, 0.4, 0.1]]])
    explainable_attention(tokens, mock_weight)

    mock_3heads = np.array([[[0.1, 0.5, 0.3, 0.1],
                              [0.2, 0.2, 0.5, 0.1],
                              [0.4, 0.2, 0.2, 0.2]]])
    print(f"1 Head  Shape (Batch,Head,Seq): {mock_weight.shape}")
    print(f"3 Heads Shape (Batch,Head,Seq): {mock_3heads.shape}")
    print(f"\n3 Heads Average")
    explainable_attention(tokens, mock_3heads)

    # Pre-Norm test
    test_prenorm()

    # Full demo
    run_demo()

    # =====================================================================
# 3.1 — Positional Encoding
# =====================================================================

# w. PE ของ pos=0 กับ pos=4 ต่างกันอย่างไรใน dimension แรก — อธิบาย pattern sin/cos
# - ใน dimension แรก (index 0 ซึ่งเป็นเลขคู่) ฟังก์ชัน SinusodalPositionEncoding จะใช้สูตร sine[cite: 3]
# - สำหรับ pos=0 ค่าที่ได้คือ sin(0) = 0[cite: 3]
# - สำหรับ pos=4 ค่าที่ได้จะเป็น sin(4 / 10000^0) = sin(4) ซึ่งจะมีค่าแตกต่างออกไปตามคลื่นความถี่[cite: 3]
# - Pattern: มิติที่เป็นเลขคู่ใช้ฟังก์ชัน sine และมิติที่เป็นเลขคี่ใช้ฟังก์ชัน cosine โดยความถี่ของคลื่นจะลดลงเรื่อยๆ เมื่อมิติ (dimension) สูงขึ้น[cite: 3]

# x. ทำไม PE ถึงใช้ทั้ง sin และ cos — ถ้าใช้ sin อย่างเดียวจะเกิดปัญหาอะไร
# - การใช้ทั้ง sin และ cos ทำให้โมเดลสามารถเรียนรู้ความสัมพันธ์ของตำแหน่งสัมพัทธ์ (Relative Position) ได้ง่ายขึ้น 
#   เพราะสำหรับระยะห่าง $k$ ใดๆ $PE_{pos+k}$ สามารถเขียนให้อยู่ในรูปผลรวมเชิงเส้น (Linear Combination) ของ $PE_{pos}$ ได้
# - หากใช้แค่ sin อย่างเดียว เวกเตอร์ของตำแหน่งต่างๆ อาจไม่สามารถครอบคลุมปริภูมิ (Space) ได้สมบูรณ์เพียงพอในการรักษาคุณสมบัติระยะห่างที่เสถียรเมื่อมีการเลื่อนตำแหน่ง

# y. แสดงผล pe.encode(X) โดยที่ X เป็น random embeddings shape (5,16) — อธิบายว่า encode() ทำอะไร
# - ฟังก์ชัน encode() ทำหน้าที่นำ Matrix ของ Positional Encoding ไป "บวก" (Add) เข้ากับ Matrix ของ Word Embeddings (X) ตรงๆ[cite: 3]
# - สมการคือ $X_{out} = X + PE$[cite: 3]
# - จุดประสงค์เพื่อแทรกข้อมูล "ลำดับและตำแหน่ง" เข้าไปในเวกเตอร์คำศัพท์ เนื่องจากกลไก Attention ของ Transformer ไม่มีอคติเรื่องลำดับคำในประโยคแต่แรก


# =====================================================================
# 3.2 — Multi-Head Attention & Shape Analysis
# =====================================================================

# z. อธิบาย weights shape (n_heads, seq, seq) — แต่ละ dimension หมายถึงอะไร
# - Dimension ที่ 1 (n_heads): จำนวนหัวของ Attention ที่ใช้ในการแบ่งการเรียนรู้มิติต่างๆ[cite: 3]
# - Dimension ที่ 2 (seq แรก): จำนวน Token หรือความยาวประโยคฝั่ง Query (คำที่กำลังมองหาความสัมพันธ์)[cite: 3]
# - Dimension ที่ 3 (seq หลัง): จำนวน Token ฝั่ง Key (คำเป้าหมายที่ Query ไปให้ความสนใจ หรือ Attention weight ที่ตกลงไป)[cite: 3]

# aa. อธิบายทุกขั้นตอนใน forward()
# 1. layer_norm: ทำการ Normalization ข้อมูล Input เพื่อปรับสเกลให้เสถียรก่อนเข้ากระบวนการ (Pre-Norm)[cite: 3]
# 2. Q,K,V projection: นำ x_norm ไปคูณกับ W_q, W_k, W_v เพื่อสร้าง Query, Key, Value vectors[cite: 3]
# 3. split_heads: แบ่ง d_model ออกเป็น n_heads ส่วน (ขนาด d_k) เพื่อให้แต่ละ Head เรียนรู้ข้อมูลแยกกัน[cite: 3]
# 4. attention: คำนวณ Scaled Dot-Product Attention (หาความสัมพันธ์ Q กับ K แล้วคูณ V)[cite: 3]
# 5. combine: นำผลลัพธ์จากทุก Head มาต่อกัน (Concatenate) กลับให้มีขนาด d_model เท่าเดิม[cite: 3]
# 6. residual: นำผลลัพธ์จาก Attention ไป "บวก" กับ Input เดิม (x) เพื่อป้องกันปัญหา Gradient Vanishing[cite: 3]

# bb. เหตุใด d_k = d_model // n_heads = 4 — ถ้า n_heads=8 จะเกิดอะไรกับ d_k
# - d_k คือขนาด dimension ย่อยของแต่ละ Head[cite: 3] การหารด้วย n_heads เพื่อรักษาให้ปริมาณการคำนวณรวม (Computational Cost) เท่ากับการใช้ Single-head ขนาด d_model 
# - หาก d_model=16 และ n_heads=8 จะได้ d_k = 16 // 8 = 2[cite: 3] (แต่ละ Head จะมีมิติข้อมูลเล็กลงเหลือ 2)

# cc. Pre-Norm คืออะไร — แตกต่างจาก Post-Norm อย่างไร และทำไม gradient ไหลดีกว่า
# - Pre-Norm คือการทำ Layer Normalization "ก่อน" ที่จะนำข้อมูลเข้า Sub-layer (เช่น MHA หรือ FFN)[cite: 3]
# - Post-Norm คือการทำหลังจากคำนวณ Sub-layer เสร็จแล้วและบวก Residual กลับเข้าไป
# - Gradient ไหลดีกว่าเพราะมีเส้นทางตรง (Direct path) ผ่าน Residual connection โดยไม่ต้องถูกขัดจังหวะจากการทำ Normalization ซ้ำๆ ทำให้เทรนโมเดลลึกๆ ได้ง่ายขึ้น


# =====================================================================
# 3.3 — Encoder vs Decoder (Causal Mask)
# =====================================================================

# dd. อธิบายความแตกต่างของ heatmap ระหว่าง Encoder กับ Decoder — token t3 มองเห็น token ใดได้บ้าง
# - Encoder: ไม่มี Mask (Full Attention) ทำให้ t3 สามารถมองเห็นได้ทุก token ทั้งอดีต ปัจจุบัน และอนาคต (t0, t1, t2, t3, t4)[cite: 3]
# - Decoder: มี Causal Mask บังอนาคตไว้ ทำให้ t3 มองเห็นได้เฉพาะ token ในอดีตและตัวมันเอง (t0, t1, t2, t3) แต่จะไม่เห็น t4[cite: 3]

# ee. Causal Mask สร้างอย่างไร — np.triu(np.ones((T,T), dtype=bool), k=1) หมายความว่าอะไร
# - ฟังก์ชัน np.triu สร้าง Upper Triangular Matrix (เมทริกซ์สามเหลี่ยมบน)[cite: 3]
# - k=1 หมายถึงการเลื่อนเส้นทแยงมุมหลักขึ้นไป 1 สเตป[cite: 3] ทำให้ตำแหน่งที่อยู่เหนือเส้นทแยงมุมหลัก (ข้อมูลในอนาคต) มีค่าเป็น True (เพื่อนำไปถูก Mask) และส่วนอื่นเป็น False[cite: 3]

# ff. ค่า -1e9 ใน causal mask ทำงานร่วมกับ softmax อย่างไร — ทำไมต้องใช้ -1e9 ไม่ใช่ 0
# - ค่า -1e9 (ค่าลบอนันต์) จะถูกนำไปแทนที่ในตำแหน่งที่โดน Mask ก่อนเข้าฟังก์ชัน Softmax[cite: 3]
# - เนื่องจาก $softmax(x) = \frac{e^x}{\sum e^x}$ เมื่อ $x = -1e9$ ค่า $e^{-1e9}$ จะเข้าใกล้ 0 ทำให้ Attention weight ของ token อนาคตเป็นศูนย์สมบูรณ์ 
# - หากใช้ค่า 0 แทน $e^0$ จะเท่ากับ 1 ซึ่งจะทำให้ token อนาคตยังมีน้ำหนักเหลืออยู่ และทำให้โมเดลขี้โกงมองเห็นอนาคตได้

# gg. งาน classification คดี IP ควรใช้ Encoder หรือ Decoder — อธิบายเหตุผลจาก output mechanism
# - ควรใช้ Encoder (เช่น BERT)[cite: 3] 
# - เหตุผล: งาน Classification จำเป็นต้องเข้าใจบริบทแบบองค์รวม (Bidirectional context) ทั้งซ้ายและขวา เพื่อสรุปความหมายของประโยคทั้งหมดออกมาเป็น Class เดียว 
#   ในขณะที่ Decoder ออกแบบมาให้คาดเดาคำถัดไปทีละคำ (Autoregressive) ซึ่งเหมาะกับงาน Text Generation มากกว่า


# =====================================================================
# 3.4 — XAI & Physics Gate
# =====================================================================

# hh. token ใดมี importance สูงสุดจาก explainable_attention() — สอดคล้องกับความหมายของประโยคหรือไม่
# - หากอ้างอิงจากตัวอย่างในโค้ด tokens "ละเมิด" และ "สิทธิบัตร" จะมีค่า Importance สูงสุด (เช่น 0.4 ใน mock data)[cite: 3]
# - สอดคล้องกับความหมายของประโยคอย่างยิ่ง เพราะทั้งสองคำคือ Key Action และ Key Entity หลักที่เป็นแก่นของการกระทำผิดในคดี IP

# ii. คำนวณ physics_score ของทุก token เมื่อ confidence=0.85 — token ใด trigger sensor (score > 0.4)
# - สมมติให้ attention ของ "ละเมิด" = 0.4 (ค่าสูงสุดจากการวิเคราะห์)
# - สูตร: score = min(2.0, base_weight * attn * confidence)
# - คำนวณ "ละเมิด": score = min(2.0, 1.2 * 0.4 * 0.85) = 0.408
# - ผลลัพธ์: Token "ละเมิด" จะ Trigger sensor เนื่องจากมีคะแนน (0.408) สูงกว่าเกณฑ์ 0.4

# jj. ทำซ้ำด้วย confidence=0.3 — ผลเปลี่ยนไปอย่างไร และทำไมถึงต้องเพิ่ม confidence เข้าสูตร
# - คำนวณใหม่: score = min(2.0, 1.2 * 0.4 * 0.3) = 0.144
# - ผลลัพธ์: คะแนนลดลงและ "ไม่" Trigger sensor อีกต่อไป
# - เหตุผลที่ต้องใช้ confidence: เพื่อป้องกัน False Alarm แม้ตัวประโยคจะมีคีย์เวิร์ดที่กระตุ้น Attention ได้สูง แต่ถ้าโมเดลโดยรวมมีความมั่นใจต่ำ (เช่น ประโยคอาจจะกำกวม) 
#   การคูณด้วย Confidence ต่ำจะช่วยกดคะแนนไว้ไม่ให้อุปกรณ์ Physical Sensor ทำงานผิดพลาด

# kk. อธิบายว่า mean(axis=0) ใน explainable_attention() ทำหน้าที่อะไร — ถ้ารวม information จาก head ต่างๆ อย่างไร
# - mean(axis=0) ทำหน้าที่ "หาค่าเฉลี่ย" ของ Attention weights จากทุกๆ Head (ในมิติที่ 0 หรือ n_heads)[cite: 3]
# - ในกรณีที่ n_heads=4 แต่ละ Head อาจจะจับใจความคนละแบบ (เช่น Head 1 ดูประธาน-กริยา, Head 2 ดูคำขยาย) 
#   การหาค่าเฉลี่ยช่วยรวมมุมมองที่หลากหลายเหล่านั้น ให้กลายเป็นคะแนนความสำคัญ (Importance score) แบบองค์รวมเพียงค่าเดียวสำหรับแต่ละคำ[cite: 3]