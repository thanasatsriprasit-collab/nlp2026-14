import numpy as np

# ============================================================
# 1. Positional Encoding
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

    def show_with_words(self, words):
        print(f"{'index':<7} | {'word':<10} | {'Positional Encoding (First 4 dims)':<40}")
        for i, word in enumerate(words):
            if i >= len(self.pe): break
            vec = self.pe[i, :4] # ตัดให้แสดงผลดูง่ายขึ้น
            vec_str = " ".join(f"{v:+.3f}" for v in vec)
            print(f"Pos_{i:<3} | {word:<10}| [{vec_str}...]")

# ============================================================
# 2. XAI & Physics Gate (จากโจทย์ 3.4)
# ============================================================
SENSOR_MAP = {
    "ผลิต":    {"sensor": "MANUFACTURING_SENSOR", "base_weight": 1.3},
    "ละเมิด":   {"sensor": "VIOLATION_MONITOR",    "base_weight": 1.2},
    "นำเข้า":   {"sensor": "CUSTOMS_GPS",          "base_weight": 1.2},
    "จำหน่าย":  {"sensor": "POS_TRACKING",         "base_weight": 1.1},
}

def physics_score(token, attn, confidence):
    cfg = SENSOR_MAP.get(token, {"base_weight": 0.5})
    score = min(2.0, cfg["base_weight"] * attn * confidence)
    return score

def explainable_attention(tokens, weights_batch, confidence=0.85):
    # หาค่าเฉลี่ยของ attention จากทุกหัว (heads)
    avg_weight = weights_batch[0].mean(axis=0) 
    print(f"\n--- XAI : Legal Importance Analysis (Confidence={confidence}) ---")
    for i, token in enumerate(tokens):
        # ดึงค่า importance สูงสุดในมิติที่คำนั้นไป focus
        importance = np.max(avg_weight[i]) 
        score = physics_score(token, importance, confidence)
        status = "🚨 TRIGGERED" if score > 0.4 else "✅ NORMAL"
        
        bar = "█" * int(importance * 20)
        print(f"{token:<12} | {bar:<20} | Attn: {importance:.2f} | Physics Score: {score:.3f} -> {status}")

# ============================================================
# Execution Block
# ============================================================
if __name__ == "__main__":
    print(">>> รันโจทย์ 3.1: Positional Encoding")
    pe = SinusodalPositionEncoding(max_seq_len=10, d_model=16)
    words = ["จำเลย", "ผลิต", "สินค้า", "ละเมิด", "สิทธิบัตร"]
    pe.show_with_words(words)

    print("\n>>> รันโจทย์ 3.4: XAI & Physics Gate")
    # จำลอง Attention Weights (Shape: Batch, Heads, Seq, Seq)
    mock_weights = np.array([[[
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.8, 0.1, 0.1, 0.1], # ให้ 'ผลิต' มีความสำคัญสูง
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.5, 0.1], # ให้ 'ละเมิด' มีความสำคัญปานกลาง
        [0.1, 0.1, 0.1, 0.1, 0.9],
    ]]])
    
    # ทดสอบกรณีมั่นใจสูง (Confidence = 0.85)
    explainable_attention(words, mock_weights, confidence=0.85)
    
    # ทดสอบกรณีมั่นใจต่ำ (Confidence = 0.3)
    explainable_attention(words, mock_weights, confidence=0.3)