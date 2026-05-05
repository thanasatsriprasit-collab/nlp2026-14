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
from collections import Counter
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ==========================================
# DATASET
# ==========================================
THAI_IP_DATASET = [
    {"text": "จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์เลขที่ 12345", "label": 0},
    {"text": "บริษัทนำเข้าชิ้นส่วนที่ละเมิดสิทธิบัตรจากต่างประเทศ", "label": 0},
    {"text": "ผู้ต้องหาทำซ้ำโปรแกรมคอมพิวเตอร์มีลิขสิทธิ์โดยไม่ได้รับอนุญาต", "label": 1},
    {"text": "บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรอย่างถูกต้องตามสัญญา", "label": 2},
    # (ย่อข้อมูลเพื่อความกระชับ สามารถนำข้อมูลเต็มจาก source ของคุณมาใส่แทนได้)
]

# ==========================================
# โจทย์ 1.1: Tokenization & Compound Protection
# ==========================================
class ThaiLegalTokenizer:
    def __init__(self, remove_stopwords=False):
        self.LEGAL_KEYWORDS = ["ละเมิดสิทธิบัตร", "เครื่องหมายการค้า", "ลิขสิทธิ์", "การกระทำความผิด"]
        self.remove_stopwords = remove_stopwords
        self.stopwords = set(thai_stopwords())

    def __call__(self, text):
        # 1. Protect Compound Keywords
        sorted_kw = sorted(self.LEGAL_KEYWORDS, key=len, reverse=True)
        placeholders = {}
        protected = text
        for i, kw in enumerate(sorted_kw):
            ph = f"__KW{i}__"
            if kw in protected:
                placeholders[ph] = kw
                protected = protected.replace(kw, ph)
                
        # 2. Tokenize ด้วย pythainlp
        tokens_raw = word_tokenize(protected, engine="newmm", keep_whitespace=False)
        
        # 3. Restore placeholder
        tokens = [placeholders.get(t, t) for t in tokens_raw]
        
        # 4. Remove Stopwords (ถ้าเปิดใช้งาน)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
            
        return tokens

# ==========================================
# โจทย์ 1.2: TF-IDF & OOV Analysis
# ==========================================
class W1Pipeline:
    def __init__(self, max_features=50, oov_strategy="unk_token", evaluate_coverage=True):
        self.tokenizer = ThaiLegalTokenizer(remove_stopwords=True)
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, token_pattern=None, max_features=max_features)
        self.evaluate_coverage = evaluate_coverage
        
    def fit_transform(self, texts):
        self.X = self.vectorizer.fit_transform(texts)
        self.vocab = self.vectorizer.get_feature_names_out()
        return self.X
        
    def print_professional_report(self, texts):
        print("\n--- W1 Pipeline Professional Report ---")
        print(f"Matrix Shape: {self.X.shape}")
        print(f"Vocabulary Size: {len(self.vocab)}")
        
        # คำนวณ Coverage
        total_tokens = 0
        known_tokens = 0
        for text in texts:
            toks = self.tokenizer(text)
            total_tokens += len(toks)
            known_tokens += sum(1 for t in toks if t in self.vocab)
            
        coverage = (known_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"Vocabulary Coverage: {coverage:.2f}%\n")

# ==========================================
# โจทย์ 1.3: Entity Extraction & Physics Gate Weight
# ==========================================
class LegalEntity:
    def __init__(self, entity_type, value, conf):
        self.entity_type = entity_type
        self.value = value
        self.conf = conf

class ThaiIPEntityExtractor:
    def __init__(self, use_context_aware=True):
        self.use_context_aware = use_context_aware
        
    def extract(self, text):
        entities = []
        conf_patent = 0.80
        conf_infringe = 0.80
        
        # Context-aware logic: เพิ่ม confidence ถ้ามีคำบอกบริบทแวดล้อม
        if self.use_context_aware:
            context_words = ["จำเลย", "ผู้ต้องหา", "เลียนแบบ", "เลขที่"]
            if any(w in text for w in context_words):
                conf_patent = min(1.0, conf_patent + 0.15)
                conf_infringe = min(1.0, conf_infringe + 0.15)
                
        # ดึง Entity
        if "สิทธิบัตร" in text:
            entities.append(LegalEntity("IP_TYPE", "PATENT", conf_patent))
        elif "ลิขสิทธิ์" in text:
            entities.append(LegalEntity("IP_TYPE", "COPYRIGHT", conf_patent))
            
        if "ละเมิด" in text or "เลียนแบบ" in text or "ทำซ้ำ" in text:
            entities.append(LegalEntity("ACTION", "INFRINGEMENT", conf_infringe))
            
        return entities

class ThaiLegalHierarchy:
    def compute_physics_gate_weight(self, entities):
        base_weight = 5.0
        for e in entities:
            if e.value == "PATENT": base_weight += 2.0
            if e.value == "COPYRIGHT": base_weight += 1.5
            if e.value == "INFRINGEMENT": base_weight += 1.5
        return min(base_weight, 10.0)

# ==========================================
# โค้ดสำหรับรันทดสอบตามรูปโจทย์
# ==========================================
if __name__ == "__main__":
    print(">>> รันโจทย์ 1.1")
    tok = ThaiLegalTokenizer(remove_stopwords=True)
    sentences_1 = [
        "จำเลยผลิตและจำหน่ายสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์",
        "บริษัทนำเข้าชิ้นส่วนที่ละเมิดสิทธิบัตรจากต่างประเทศ",
        "ผู้ต้องหาทำซ้ำโปรแกรมคอมพิวเตอร์มีลิขสิทธิ์โดยไม่ได้รับอนุญาต"
    ]
    for s in sentences_1:
        print(f"Tokens: {tok(s)}")

    print("\n>>> รันโจทย์ 1.2")
    texts = [d["text"] for d in THAI_IP_DATASET]
    w1 = W1Pipeline(max_features=50, oov_strategy="unk_token", evaluate_coverage=True)
    X = w1.fit_transform(texts)
    w1.print_professional_report(texts)

    print(">>> รันโจทย์ 1.3")
    extractor = ThaiIPEntityExtractor(use_context_aware=True)
    hierarchy = ThaiLegalHierarchy()
    for d in THAI_IP_DATASET[:4]: # ทดสอบ 4 ตัวอย่างแรก
        entities = extractor.extract(d["text"])
        weight = hierarchy.compute_physics_gate_weight(entities)
        entity_types = [e.entity_type for e in entities]
        print(f"Label={d['label']} weight={weight:.2f} entities={entity_types}")