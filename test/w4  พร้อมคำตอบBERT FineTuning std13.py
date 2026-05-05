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

MODEL_REGISTRY = {
    "mbert": {
        "hf_name" : "bert-base-multilingual-cased",
        "params"  : "110M",
        "thai_cov": "~1%",
        "note"    : "ใช้ได้หลายภาษา แต่ Thai coverage น้อย",
    },
    "xlmr": {
        "hf_name" : "xlm-roberta-base",
        "params"  : "270M",
        "thai_cov": "~5%",
        "note"    : "RoBERTa architecture, Thai ดีกว่า mBERT",
    },
    "wangchanberta": {
        "hf_name" : "airesearch/wangchanberta-base-att-spm-uncased",
        "params"  : "110M",
        "thai_cov": "100%",
        "note"    : "ดีที่สุดสำหรับ Thai legal text",
    },
}

def show_model_comparison():
    print("=" * 60)
    print("  4.1 เลือก Pretrained Model สำหรับ Thai Legal Text")
    print("=" * 60)
    print(f"  {'Model':<16} {'Params':<8} {'Thai%':<8} หมายเหตุ")
    print(f"  {'─'*56}")
    for name, m in MODEL_REGISTRY.items():
        print(f"  {name:<16} {m['params']:<8} {m['thai_cov']:<8} {m['note']}")
    print(f"\n  → เลือก WangchanBERTa เพราะ pre-train บน Thai Wikipedia + CCNet")

show_model_comparison()

def get_device():
    """ตรวจสอบ GPU อัตโนมัติ: CUDA → MPS → CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif torch.backends.mps.is_available():
            print(f"  ✅ Apple Silicon (MPS)")
            return "mps"
        else:
            print(f"  ⚠️  ไม่มี GPU → ใช้ CPU (ช้ากว่า 10-20x)")
            return "cpu"
    except ImportError:
        print("  ℹ️  ไม่มี PyTorch → ใช้ Mock mode")
        return "mock"

DEVICE = get_device()
print(f"\n  Device ที่ใช้: {DEVICE}")


class MockTokenize:
    """จำลอง BERT Tokenizer"""
    LEGAL_TERMS = ["ภูมิปัญญาท้องถิ่น", "สิทธิการประดิษฐ์", "อนุสิทธิบัตร", "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิ"]

    def encode(self, texts, max_length=128):
        """แปลงข้อความ -> input_ids + attention_mask
           input_ids : [CLS] + token_ids + [SEP] + [PAD]...
           attention_mask: 1 = real token, 0 = padding
        """
        if isinstance(texts, str):
            texts = [texts]

        input_ids, attention_mask = [], []  # ✅ reset ก่อน loop
        for text in texts:
            # [CLS=1] ... [SEP=2]
            ids = [1] + [ord(c) % 5000 + 100 for c in text[:max_length - 2]] + [2]  # ✅ แก้ typo และ bracket
            pad_len = max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids  = ids + [0] * pad_len
            input_ids.append(ids)           # ✅ เพิ่ม append ที่หายไป
            attention_mask.append(mask)     # ✅ เพิ่ม append ที่หายไป

        return {
            "input_ids"      : np.array(input_ids,      dtype=np.int32),
            "attention_mask" : np.array(attention_mask, dtype=np.int32),
        }

    def show(self, text, max_length=32): # ← เพิ่ม max_length ให้ยาวพอ
        enc = self.encode([text], max_length)
        ids =enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        real_len = mask.sum()
        print(f"\n ข้อความ : '{text}'")
        print(f" input_ids : {ids.tolist()}") # ← แสดงทั้ง array รวม 0
        print(f" mask : {mask.tolist()}")
        print(f" real tokens: {real_len} PAD: {max_length - real_len}")
        print(f"\n แยกส่วน:")
        print(f" [CLS] = {ids[0]}")
        print(f" tokens= {ids[1:real_len-1].tolist()}")
        print(f" [SEP] = {ids[real_len-1]}")
        print(f" [PAD] = {ids[real_len:].tolist()}") # ← แสดง 0s


# ── Vocabulary Expansion Demo ──────────────────────────────────
def vocab_expansion_demo():
    tok = MockTokenize()
    print("=" * 58)
    print(" STEP 1: ก่อน expansion — คำใหม่ถูกตัดเป็นชิ้น")
    print("=" * 58)
    tok.show("สิทธิบัตรการประดิษฐ์")
    print("\n" + "=" * 58)
    print(" STEP 2: เพิ่มคำใหม่เข้า vocab (Vocabulary Expansion)")
    print("=" * 58) # vocab ปกติ + เพิ่มคำกฎหมาย
    base_vocab_size = 5000
    new_vocab = {term:base_vocab_size + i
             for i, term in enumerate(MockTokenize.LEGAL_TERMS)}
    print(f"\n vocab เดิม : {base_vocab_size} คำ")
    print(f" เพิ่มคำใหม่: {len(new_vocab)} คำ")
    print(f" vocab ใหม่ : {base_vocab_size +len(new_vocab)} คำ\n")
    for term, idx in new_vocab.items():
        print(f" '{term}' → id {idx} (token ใหม่ weight = random ❗)")
    print("\n" + "=" * 58)
    print(" STEP 3: ทำไมต้อง Warm-up ก่อน Fine-tune")
    print("=" * 58)
    print(""" ปัญหา: คำเดิม → weights ผ่าน pre-train มาแล้ว (มีความหมาย) คำใหม่ → weights = random ❗ (ยังไม่มีความหมาย) ถ้า fine-tune ทุก layer พร้อมกันเลย: gradient จากคำใหม่ (random) จะรบกวน weights เดิม → โมเดลลืมสิ่งที่เรียนมา = Catastrophic Forgetting ❌ วิธีแก้ — Warm-up 3 ขั้นตอน: ขั้น 1 │ Freeze ทุก layer ยกเว้น embedding │ train แค่ embedding 2-3 epochs│ → คำใหม่เริ่มมีความหมาย ขั้น 2 │ Unfreeze ทุก layer │ train ด้วย LR ต่ำ (2e-5) │ → ปรับ weights ทั้งหมดพร้อมกันขั้น 3 │ Fine-tune จนกว่า val_loss นิ่ง │ → โมเดลพร้อมใช้งาน ✅ """)
    print("=" * 58)
    print(" STEP 4: จำลอง embedding weight ก่อน/หลัง warm-up")
    print("=" * 58)
    np.random.seed(42)
    d_model = 8 # embedding ของคำเดิม (pretrained) — มีค่าชัดเจน
    old_emb = np.array([0.82, -0.34, 0.56, 0.91, -0.12, 0.67, -0.45, 0.23]) # embedding ของคำใหม่ก่อน warm-up — random
    new_before = np.random.randn(d_model) * 0.02 # หลัง warm-up — เริ่มมีทิศทาง (จำลอง)
    new_after = old_emb *0.6 + np.random.randn(d_model) * 0.1
    print(f"\n คำเดิม 'ละเมิด' : {old_emb.round(2)}")
    print(f" คำใหม่ ก่อน warm-up : {new_before.round(2)} ← random")
    print(f" คำใหม่ หลัง warm-up : {new_after.round(2)} ← มีทิศทางแล้ว")

    # cosine similarity
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"\n cosine similarity กับ 'ละเมิด':")
    print(f" ก่อน warm-up : {cosine(old_emb, new_before):.3f} (ไม่เกี่ยวกัน)")
    print(f" หลัง warm-up : {cosine(old_emb, new_after):.3f} (ใกล้เคียงกัน)")
   
    # ── รัน ──────────────────────────────────────────────────────
tok = MockTokenize()
tok.show("ผู้ต้องหาละเมิดสิทธิบัตร")
print()
vocab_expansion_demo()
 
# 4. BERT Encoder
import numpy as np

# --- 1. Mock Tokenizer (สำหรับจำลองการแปลงข้อความ) ---
class MockTokenizer:
    def encode(self, texts, max_length=32):
        batch_size = len(texts)
        # จำลองการสร้าง input_ids และ attention_mask
        input_ids = np.random.randint(100, 10000, size=(batch_size, max_length))
        attention_mask = np.ones((batch_size, max_length))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

# --- 2. Mock BERT Encoder (จำลอง Transformer 12 Layers) ---
class MockBERTEncoder:
    """
    จำลอง BERT 12-layer encoder
    Output: h_cls shape (batch, 768)
    """
    def __init__(self, hidden_size=768, seed=42):
        self.hidden_size = hidden_size
        self.rng = np.random.RandomState(seed)

    def forward(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        # จำลองค่า h_cls (Hidden state ของ [CLS] token)
        h_cls = self.rng.randn(batch, self.hidden_size).astype(np.float32) * 0.1
        return h_cls

# --- 3. Classification Head (The Judge) ---
class ClassificationHead:
    """
    Linear layer สำหรับจำแนก class (PATENT, COPYRIGHT, NONE)
    """
    def __init__(self, hidden_size=768, n_classes=3, dropout=0.1, seed=42):
        rng = np.random.RandomState(seed)
        # Xavier Initialization สำหรับน้ำหนัก
        s = np.sqrt(2.0 / (hidden_size + n_classes))
        self.W = rng.randn(n_classes, hidden_size).astype(np.float32) * s
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.dropout = dropout

    def forward(self, h_cls, training=False):
        # Dropout: ใช้เฉพาะช่วง training เพื่อป้องกัน Overfitting
        if training:
            mask = (np.random.rand(*h_cls.shape) > self.dropout).astype(np.float32)
            h_cls = h_cls * mask / (1 - self.dropout)

        # คำนวณ Logits: (batch, 768) @ (768, 3) = (batch, 3)
        logits = h_cls @ self.W.T + self.b
        
        # Softmax: แปลงเป็นความน่าจะเป็น
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

# --- 4. BERT For Classification (Main Class) ---
class BERTForClassification:
    CLASS_NAMES = ["ละเมิด_สิทธิบัตร", "ละเมิด_ลิขสิทธิ์", "ไม่ละเมิด"]

    def __init__(self, seed=42):
        self.encoder = MockBERTEncoder(seed=seed)
        self.head = ClassificationHead(seed=seed)
        self.tokenizer = MockTokenizer()

    def predict_proba(self, input_ids, attention_mask):
        h_cls = self.encoder.forward(input_ids, attention_mask)
        return self.head.forward(h_cls)

    def predict(self, input_ids, attention_mask):
        return np.argmax(self.predict_proba(input_ids, attention_mask), axis=1)

    def show_prediction(self, texts):
        # ขั้นตอน Tokenization
        enc = self.tokenizer.encode(texts, max_length=32)
        
        # ขั้นตอนการทำนาย
        prob = self.predict_proba(enc["input_ids"], enc["attention_mask"])
        pred = np.argmax(prob, axis=1)

        # แสดงผลลัพธ์
        print(f"\n{'ข้อความ':<45} {'การทำนาย':<20} {'ความมั่นใจ'}")
        print(f"{'─'*75}")
        for text, p, pr in zip(texts, pred, prob):
            confidence = pr[p] * 100
            print(f"{text[:43]:<45} {self.CLASS_NAMES[p]:<20} {confidence:.1f}%")

# --- 5. Execution Block ---
if __name__ == "__main__":
    # เริ่มต้นโมเดล
    model = BERTForClassification(seed=42)
    
    # ตัวอย่างข้อมูลทดสอบ (Legal Context)
    test_cases = [
        "ผู้ต้องหานำเข้าสินค้าปลอมแปลงสิทธิบัตร",
        "จำเลยทำซ้ำงานที่มีลิขสิทธิ์โดยไม่ได้รับอนุญาต",
        "บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรถูกต้องแล้ว",
        "มีการดัดแปลงโปรแกรมคอมพิวเตอร์เพื่อการค้า"
    ]
    
    model.show_prediction(test_cases)

# =====================================================================
# 4.1 — Model Selection & Tokenization
# =====================================================================

# ll. อธิบาย input_ids structure: [CLS]=1, token_ids, [SEP]=2, [PAD]=0 — แต่ละส่วนมีบทบาทอะไรใน BERT
# - [CLS] (Classify) = 1: เป็น token แรกสุดของทุก Sequence ทำหน้าที่เป็น "ตัวแทนของทั้งประโยค"[cite: 4]
#   เวกเตอร์ของ [CLS] ในเลเยอร์สุดท้ายจะถูกดึงไปใช้ในงาน Classification
# - token_ids: คือตัวเลข ID ที่เป็นตัวแทนของแต่ละคำหรือ subword ในประโยคที่ถูกตัดแล้ว[cite: 4]
# - [SEP] (Separator) = 2: ใช้คั่นระหว่างประโยค (หากมี 2 ประโยค) หรือใช้บอก "จุดสิ้นสุด" ของประโยค[cite: 4]
# - [PAD] (Padding) = 0: ใช้เติมช่องว่างให้เต็ม max_length เพื่อให้ข้อมูลใน Batch เดียวกันมีความยาวเท่ากัน[cite: 4]

# mm. attention_mask=1 vs 0 ต่างกันอย่างไร — BERT ใช้ mask นี้ตรงไหนใน self-attention
# - 1: คือ Real token (คำที่มีอยู่จริงในประโยครวมถึง CLS และ SEP) ซึ่งโมเดล "ต้อง" ให้ความสนใจ (Pay attention)[cite: 4]
# - 0: คือ Padding token ซึ่งเป็นแค่ตัวเติมเต็ม โมเดล "ไม่ต้อง" ให้ความสนใจ[cite: 4]
# - การใช้ใน Self-Attention: BERT จะนำ attention_mask ไปแปลงเป็น -1e9 หรือค่าลบมากๆ (คล้าย Causal Mask แต่บังแค่ PAD) 
#   เพื่อใส่ในตำแหน่งที่เป็น 0 ก่อนเข้า Softmax ทำให้ Attention weight ที่ตกกับ Padding กลายเป็น 0[cite: 4]

# nn. ทำไม WangchanBERTa ถึงเหมาะกว่า mBERT สำหรับ dataset นี้ — เปรียบเทียบ Thai coverage 1% vs 100%
# - mBERT แม้จะรองรับหลายภาษา แต่มีพื้นที่คลังคำศัพท์ (Coverage) ภาษาไทยน้อยมาก (ประมาณ 1%)[cite: 4] 
#   ทำให้คำไทยส่วนใหญ่กลายเป็น Unknown หรือถูกหั่นเป็น subword เล็กๆ ยิบย่อย (Fragmentation) ทำให้สูญเสียความหมายทางกฎหมาย
# - WangchanBERTa ถูก Pre-train มาเพื่อภาษาไทยโดยเฉพาะ (100% Thai Coverage บน Wiki+CCNet)[cite: 4]
#   จึงเข้าใจโครงสร้าง ไวยากรณ์ และคำศัพท์เฉพาะทางของไทยได้ดีกว่ามาก เหมาะสมที่สุดสำหรับ Thai Legal Text[cite: 4]


# =====================================================================
# 4.2 — Vocabulary Expansion & Warm-up
# =====================================================================

# oo. อธิบาย Catastrophic Forgetting — เกิดขึ้นได้อย่างไรเมื่อเพิ่มคำใหม่แล้ว fine-tune ทันที
# - เมื่อเพิ่มคำศัพท์ใหม่ (เช่น "สิทธิบัตรการประดิษฐ์") น้ำหนัก (Weights) ของคำเหล่านั้นจะถูก "สุ่ม" (Random initialization) ขึ้นมาใหม่[cite: 4]
# - หาก Fine-tune ทุกเลเยอร์ทันที Gradient (ความชัน) ขนาดใหญ่จากคำศัพท์ใหม่ที่ยังเดาสุ่มอยู่ จะไหลย้อนกลับไปกระทบ Weights ของคำศัพท์เดิมที่โมเดลเรียนรู้มาดีแล้วจากช่วง Pre-train
# - ส่งผลให้โมเดล "ลืม" ความรู้เดิมที่มีอยู่ (Catastrophic Forgetting) และประสิทธิภาพพังทลายลง[cite: 4]

# pp. Warm-up 3 ขั้นตอนคืออะไร — ทำไมขั้น 2 ต้องใช้ LR=2e-5 ไม่ใช่ 1e-3
# - ขั้น 1: Freeze เลเยอร์ทั้งหมด ยกเว้น Embedding layer แล้วเทรนเพื่อให้ "คำใหม่เริ่มมีทิศทางและความหมาย" (จัดเรียงตัวใน Vector space)[cite: 4]
# - ขั้น 2: Unfreeze เลเยอร์ทั้งหมด และเทรนด้วย Learning Rate (LR) ที่ต่ำ (เช่น 2e-5)[cite: 4]
#   * เหตุผลที่ใช้ LR ต่ำ (2e-5) แทนสูง (1e-3): เพราะน้ำหนักเดิมดีอยู่แล้ว การปรับด้วย LR ต่ำๆ จะเป็นการ "ค่อยๆ จูน" (Fine-tuning) 
#   หากใช้ 1e-3 ซึ่งใหญ่เกินไป จะทำให้ค่า Weights กระโดดแรงและทำลายความรู้เดิมที่สะสมมา
# - ขั้น 3: Fine-tune ต่อไปจนกว่าค่า Validation Loss จะนิ่ง (Converge)[cite: 4]

# qq. cosine similarity ก่อน warm-up ≈ 0.38 และหลัง ≈ 0.96 หมายความว่าอะไร — อธิบายใน vector space
# - ก่อน warm-up (0.38): เวกเตอร์ของคำใหม่ (สุ่มมา) มีทิศทางกระจัดกระจาย ไม่ชี้ไปทางเดียวกับกลุ่มคำที่ความหมายคล้ายกัน (เช่น "ละเมิด")[cite: 4]
# - หลัง warm-up (0.96): ค่าเข้าใกล้ 1 หมายความว่าใน Vector space เวกเตอร์ของคำใหม่ได้ถูกปรับทิศทาง (Aligned) 
#   ให้ไปอยู่ใกล้ชิดและชี้ไปทางเดียวกับคำศัพท์ที่มีบริบทคล้ายคลึงกันทางกฎหมายแล้ว[cite: 4]

# rr. เสนอคำศัพท์กฎหมาย IP ที่ควรเพิ่มเข้า vocab อีกอย่างน้อย 5 คำ พร้อมให้เหตุผล
# 1. "ทรัพย์สินทางอุตสาหกรรม" -> เป็นหมวดหมู่ใหญ่ที่ครอบคลุมสิทธิบัตรและเครื่องหมายการค้า
# 2. "ความลับทางการค้า" -> เป็นประเภทหนึ่งของทรัพย์สินทางปัญญาที่มีคดีความบ่อย
# 3. "สิ่งบ่งชี้ทางภูมิศาสตร์" (GI) -> เป็นทรัพย์สินทางปัญญาที่สำคัญของสินค้าท้องถิ่นไทย
# 4. "ค่าสินไหมทดแทน" -> คำสำคัญที่มักปรากฏในคำพิพากษาเมื่อมีการละเมิด
# 5. "ศาลทรัพย์สินทางปัญญาและการค้าระหว่างประเทศ" -> ชื่อเฉพาะของศาลที่มีอำนาจพิจารณาคดีประเภทนี้โดยตรง


# =====================================================================
# 4.3 — BERT Classification & Analysis
# =====================================================================

# ss. อธิบาย architecture: MockBERTEncoder → h_cls (768 dim) → ClassificationHead → softmax — แต่ละส่วนทำอะไร
# - MockBERTEncoder: ทำหน้าที่รับ input_ids และ attention_mask ผ่าน Transformer layers เพื่อดึงบริบทของประโยค[cite: 4]
# - h_cls (768 dim): คือ Output vector จาก token [CLS] ซึ่งทำหน้าที่บีบอัดใจความทั้งหมดของประโยคให้อยู่ในเวกเตอร์ขนาด 768 มิติ[cite: 4]
# - ClassificationHead: เป็น Linear Layer (Fully Connected) ที่รับ h_cls มาคูณกับ Weights เพื่อแปลงจาก 768 มิติ ให้เหลือ 3 มิติ (เท่ากับจำนวน Class: ละเมิดสิทธิบัตร, ละเมิดลิขสิทธิ์, ไม่ละเมิด)[cite: 4]
# - Softmax: แปลงค่า Logits 3 มิตินั้น ให้อยู่ในรูปของ "ความน่าจะเป็น" (Probability) ที่รวมกันได้ 1.0 (100%) เพื่อใช้ทำนาย Class[cite: 4]

# tt. ทำไม BERT ใช้ [CLS] token เป็นตัวแทนของทั้งประโยค — เชื่อมกับ mean pooling ใน MiniBERT ของ W3
# - BERT ใช้ [CLS] (ตำแหน่งแรก) เพราะในกระบวนการ Self-Attention แบบ Bidirectional นั้น [CLS] สามารถมองเห็นและเก็บเกี่ยวความสัมพันธ์จากทุกๆ Token ในประโยคได้เท่าเทียมกันตั้งแต่ต้นจนจบ
# - ต่างจาก MiniBERT ใน W3 ที่อาจใช้ Mean Pooling (นำเวกเตอร์ทุกคำมาหาค่าเฉลี่ย) การใช้ [CLS] เป็นจุดสนใจเดียว (Single bottleneck) ช่วยให้โมเดลมีพื้นที่เฉพาะสำหรับเรียนรู้ "สัดส่วนสำคัญ" (Global representation) เพื่อการจำแนกโดยเฉพาะ

# uu. Dropout ใน ClassificationHead ทำงานอย่างไร — training=True กับ training=False ต่างกันอย่างไร
# - Dropout ทำหน้าที่สุ่ม "ปิด" (ให้ค่าเป็น 0) บาง Node ในระหว่างการเทรน เพื่อบังคับไม่ให้โมเดลพึ่งพา Node ใด Node หนึ่งมากเกินไป ช่วยลดปัญหา Overfitting[cite: 4]
# - training=True: Dropout จะทำงาน มีการสุ่มปิด Node ตามเรต (เช่น 0.1 หรือ 10%) และมีการขยายสเกลค่าที่เหลือ (scaling) เพื่อชดเชย Node ที่หายไป[cite: 4]
# - training=False (ตอนนำไปใช้จริง/Predict): Dropout จะ "ปิดใช้งาน" ทุก Node ทำงานเต็ม 100% เพื่อให้ผลลัพธ์มีความเสถียรและแม่นยำที่สุด[cite: 4]

# vv. เปรียบเทียบ performance W2 (BiLSTM) กับ W4 (BERT) — BERT ได้เปรียบในด้านใด และ MockBERTEncoder ขาดอะไรจาก BERT จริง
# - ความได้เปรียบของ BERT: BERT มีกลไก Self-Attention ที่ซับซ้อน (Multi-head, Multi-layer) ทำให้เข้าใจ Long-range dependency และบริบทสองทิศทางได้ลึกซึ้งกว่า BiLSTM ที่ต้องส่งข้อมูลตามลำดับเวลา (Time-step)
# - สิ่งที่ Mock ขาดหายไป: MockBERTEncoder ในโค้ด[cite: 4] เป็นเพียงการสุ่มค่า (Random generation) h_cls ออกมาดื้อๆ[cite: 4] ขาดกลไก Transformer แท้จริง (ไม่มี Q, K, V, Multi-head attention, FFN)

# ww. ถ้าจะนำโมเดล W4 ไปใช้จริงในศาลทรัพย์สินทางปัญญา ต้องทำอะไรเพิ่มเติมอีกอย่างน้อย 3 ขั้นตอน
# 1. เปลี่ยน MockBERTEncoder เป็น Pre-trained Model ของจริง: โหลด WangchanBERTa ผ่าน Hugging Face (transformers library) มาใช้งานแทนคลาสจำลอง[cite: 4]
# 2. เตรียมข้อมูลและ Fine-tuning แท้จริง: รวบรวมชุดข้อมูลคำพิพากษาศาลทรัพย์สินทางปัญญาจำนวนมาก ทำ Data Cleaning และเข้าสู่กระบวนการ Warm-up / Fine-tune อย่างเต็มรูปแบบ
# 3. Model Evaluation & Explainability (XAI): ต้องวัดผลด้วย Metrics เชิงลึก (Precision, Recall, F1 ในระดับ Minority class) และใช้เทคนิค XAI (เช่น Attention visualization เหมือนใน W3) เพื่อให้ศาลสามารถ "อธิบายได้" ว่าทำไมโมเดลถึงตัดสินว่าละเมิด