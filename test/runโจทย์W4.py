import numpy as np

# ============================================================
# 1. Model Registry & Vocabulary Expansion
# ============================================================
class MockTokenizer:
    LEGAL_TERMS = ["ภูมิปัญญาท้องถิ่น", "สิทธิการประดิษฐ์", "อนุสิทธิบัตร", "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิ"]

    def encode(self, texts, max_length=32):
        if isinstance(texts, str): texts = [texts]
        input_ids, attention_mask = [], []
        
        for text in texts:
            # จำลอง Tokenization: [CLS=1] ... [SEP=2] ... [PAD=0]
            ids = [1] + [ord(c) % 5000 + 100 for c in text[:max_length - 2]] + [2]
            pad_len = max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [0] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)

        return {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
        }

def vocab_expansion_demo():
    print(">>> Vocabulary Expansion Demo")
    MY_LEGAL_TERMS = ["อนุสิทธิบัตร", "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิ", "ความลับทางการค้า", "ค่าสินไหมทดแทน"]
    base_vocab = 5000
    new_vocab = {t: base_vocab + i for i, t in enumerate(MY_LEGAL_TERMS)}
    print(f"เพิ่มคำศัพท์ใหม่เข้า Vocabulary สำเร็จ: \n{new_vocab}\n")

# ============================================================
# 2. Mock BERT Encoder & Classification Head
# ============================================================
class MockBERTEncoder:
    def __init__(self, hidden_size=768, seed=42):
        self.hidden_size = hidden_size
        self.rng = np.random.RandomState(seed)

    def forward(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        # จำลองการสร้างเวกเตอร์ [CLS] ขนาด 768 มิติ
        h_cls = self.rng.randn(batch, self.hidden_size).astype(np.float32) * 0.1
        return h_cls

class ClassificationHead:
    def __init__(self, hidden_size=768, n_classes=3, dropout=0.1, seed=42):
        rng = np.random.RandomState(seed)
        s = np.sqrt(2.0 / (hidden_size + n_classes))
        self.W = rng.randn(n_classes, hidden_size).astype(np.float32) * s
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.dropout = dropout

    def forward(self, h_cls, training=False):
        if training:
            mask = (np.random.rand(*h_cls.shape) > self.dropout).astype(np.float32)
            h_cls = h_cls * mask / (1 - self.dropout)
        logits = h_cls @ self.W.T + self.b
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

# ============================================================
# 3. Full BERT For Classification Pipeline
# ============================================================
class BERTForClassification:
    CLASS_NAMES = ["ละเมิดสิทธิบัตร", "ละเมิดลิขสิทธิ์", "ไม่ละเมิด"]

    def __init__(self, seed=42):
        self.encoder = MockBERTEncoder(seed=seed)
        self.head = ClassificationHead(seed=seed)
        self.tokenizer = MockTokenizer()

    def predict_proba(self, input_ids, attention_mask):
        h_cls = self.encoder.forward(input_ids, attention_mask)
        return self.head.forward(h_cls)

    def show_prediction(self, texts):
        enc = self.tokenizer.encode(texts, max_length=32)
        prob = self.predict_proba(enc["input_ids"], enc["attention_mask"])
        pred = np.argmax(prob, axis=1)

        print(f"\n{'ข้อความ':<40} | {'การทำนาย':<15} | {'ความมั่นใจ'}")
        print("-" * 75)
        for text, p, pr in zip(texts, pred, prob):
            print(f"{text[:38]:<40} | {self.CLASS_NAMES[p]:<15} | {pr[p]*100:.1f}%")

# ============================================================
# Execution Block (ตรงตามโจทย์ 4.3)
# ============================================================
if __name__ == "__main__":
    vocab_expansion_demo()

    print(">>> รันโจทย์ 4.3: BERT Classification & Analysis")
    model = BERTForClassification(seed=42)
    
    # ดึงข้อมูลมาจาก Dataset จำลอง
    THAI_IP_DATASET = [
        {"text": "ผู้ต้องหานำเข้าสินค้าปลอมแปลงสิทธิบัตร"},
        {"text": "จำเลยทำซ้ำงานที่มีลิขสิทธิ์โดยไม่ได้รับอนุญาต"},
        {"text": "บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรถูกต้องแล้ว"},
        {"text": "มีการดัดแปลงโปรแกรมคอมพิวเตอร์เพื่อการค้า"},
        {"text": "ผู้ผลิตจ่ายค่าตอบแทนสิทธิเรียบร้อยแล้ว"}
    ]
    all_texts = [d["text"] for d in THAI_IP_DATASET]
    
    # 1. แสดงผลการทำนาย
    model.show_prediction(all_texts)
    
    # 2. วิเคราะห์ Confidence Distribution
    enc = model.tokenizer.encode(all_texts, max_length=32)
    prob = model.predict_proba(enc["input_ids"], enc["attention_mask"])
    conf = prob.max(axis=1) # ดึงค่าความมั่นใจสูงสุดของแต่ละเคส
    
    print("\n--- Confidence Analysis ---")
    print(f"Mean confidence: {conf.mean():.3f}")
    print(f"Low confidence (< 0.5): {(conf < 0.5).sum()} cases")