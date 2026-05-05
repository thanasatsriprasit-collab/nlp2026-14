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

import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import classification_report, confusion_matrix
import torch 
import torch.nn as nn

#  1. สร้างพจนานุกรมคำพ้องความหมาย และ Data Augmentation
LEGAL_SYNONYMS = {
"ละเมิด":["ฝ่าฝืน","กระทำผิด","ล่วงสิทธิ"],
"จำหน่าย":["ขาย","เผยแพร่","กระจายสินค้า"],
"ปลอมแปลง":["ทำเทียม","เลียนแบบ"]
}

def augment_legal_text(text):
    words = text.split()
    new_words = words.copy()
    for i , word in enumerate(words):
        if word in LEGAL_SYNONYMS:
            new_words[i] = random.choice(LEGAL_SYNONYMS[word])
    return " ".join(new_words)
original = "จำเลย ละเมิด และ จำหน่าย สินค้า"
augmented = augment_legal_text(original)
# print(f"---Data Augmentation---")
# print(f"Original : {original}")
# print(f"Augmented : {augmented}")

# 2. SMOTE with Fallback
import numpy as np
from imblearn.over_sampling import SMOTE , RandomOverSampler
from collections import Counter
def balance_legal_data(X,y):
   counts = Counter(y)
#    print(f"Original distribution : {counts}")
   # คลาสน้อยสุดมีกี่ตัว
   min_samples = min(counts.values())
   if min_samples > 1 :
       sampler = SMOTE(k_neighbors=min(5, min_samples-1),random_state=42)
   else:
       sampler = RandomOverSampler(random_state=42)
   X_res , y_res = sampler.fit_resample(X,y)
   print(f"Balanced distribution: {Counter(y_res)}")
   return X_res , y_res

# จำลองข้อมูล Imbalance (class 0 = 10 , class 1 = 2)
X_mock = np.random.randn(12,5)
y_mock = np.array([0]*10 + [1]*2)
X_res, y_res = balance_legal_data(X_mock,y_mock)

# เปลี่ยนข้อมูลนำเข้า เป็น 3  มิติ เพื่อ LSTM
X_res_tensor = torch.tensor(X_res, dtype=torch.float32)
X_res_3d = X_res_tensor.unsqueeze(1)
y_res_tensor = torch.tensor(y_res, dtype=torch.long)
print(f"Shape หลังจากทำ SMOTE (2D) : {X_res.shape}")
print(f"Shape สำหรับนำเข้า (3D) : {X_res_3d.shape}") # (Batch , Seq_len , Input_size)


    
# 3.1 BiLSTM 
class LegalBiLSTM(nn.Module):
    def __init__(self,input_dim=5, hidden_dim=32,output_dim=3):
        super(LegalBiLSTM,self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2,output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Mean Pooling
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)
model = LegalBiLSTM()
sample_input = torch.randn(1,5,5) # 1 doc 5 token 16 dim
output = model(sample_input)
# print(f"--BiLSTM Output--")
# print(f"Logics: {output.detach().numpy()}")


#3.2 LSTM
class LegalLSTM(nn.Module):
    def __init__(self,input_dim=5, hidden_dim=32,output_dim=3):
        super(LegalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim,output_dim)
        nn.init.xavier_uniform(self.fc.weight) # เลือกค่า weight ที่เหมาะสม ในการเทรนรอบแรก

    def forward(self,x):
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)
    
# 3.3 เทรนโมเดล LSTM VS BiLSTM
def train_and_evaluate(model_class , name, X,y,Class_names):
    # print(f"Training : {name} ....")
    model = model_class()
    # Cost-Sensitive Weight (FN = False Negative)
    #0 ไม่ผิด 1 ละเมิดสิทธิบัตร 2 ละเมิดลิขสิทธิ์
    weights = torch.tensor([1.0,2.0,2.0])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Sample Training Loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs= model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    # Evaluate with CM
        model.eval()
        with torch.no_grad():
            logits = model(X)
            y_pred = torch.argmax(logits, dim=1).numpy()
    cm = confusion_matrix(y,y_pred)

    # Heat Map
    fig,ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, fmt='d',cmap="Greens",
                xticklabels=Class_names,yticklabels=Class_names, ax=ax)
    ax.set_title(f"Confusion Matrix : {name}")
    ax.set_ylabel(f"Actual")
    ax.set_xlabel(f"Predicted")
    plt.tight_layout()
    plt.show()
    return classification_report(y,y_pred ,target_names=Class_names)

# รันเปรียบเทียบ Training & Evaluate
class_list = ['NO-INF', 'PATENT', 'COPYRIGHT']
# X_train , y_train = ไม่ผ่าน MOTE , X_res, y_res = ผ่านการ SMOTE แล้ว
report_lstm = train_and_evaluate(LegalLSTM,"Unidirectional LSTM", X_res_3d,y_res_tensor,class_list)
# report_bilstm = train_and_evaluate(LegalBiLSTM,"Bidirectional LSTM", X_res_3d,y_res_tensor,class_list)

# =====================================================================
# 2.1 — Data Augmentation
# =====================================================================

# k. แสดงคู่ original -> augmented อย่างน้อย 5 คู่
# 1. "จำเลย ละเมิด สิทธิบัตร" -> "จำเลย ฝ่าฝืน สิทธิบัตร" (ละเมิด -> ฝ่าฝืน)
# 2. "บริษัท จำหน่าย สินค้า" -> "บริษัท ขาย สินค้า" (จำหน่าย -> ขาย)
# 3. "ผู้ต้องหา ปลอมแปลง เอกสาร" -> "ผู้ต้องหา ทำเทียม เอกสาร" (ปลอมแปลง -> ทำเทียม)
# 4. "กระทำการ ละเมิด ลิขสิทธิ์" -> "กระทำการ ล่วงสิทธิ ลิขสิทธิ์" (ละเมิด -> ล่วงสิทธิ)
# 5. "ห้าม จำหน่าย โดยเด็ดขาด" -> "ห้าม เผยแพร่ โดยเด็ดขาด" (จำหน่าย -> เผยแพร่)

# l. ประโยคที่ไม่มีคำใน LEGAL_SYNONYMS จะเกิดอะไรขึ้น
# - ฟังก์ชันจะทำการข้ามคำนั้นไปโดยไม่มีการแก้ไขใดๆ ส่งผลให้ augmented text ที่ได้จะเหมือนกับ original text ทุกประการ

# m. เพิ่ม synonym ใหม่จำน้อย 2 คำที่เหมาะกับ dataset นี้
# - "ทำซ้ำ": ["คัดลอก", "ก๊อปปี้", "จำลอง"] -> คดีละเมิดลิขสิทธิ์มักเกี่ยวกับการทำซ้ำ การเพิ่มคำเหล่านี้ช่วยให้โมเดลเข้าใจบริบทได้กว้างขึ้น
# - "ผู้ต้องหา": ["จำเลย", "ผู้กระทำผิด", "ผู้ถูกกล่าวหา"] -> ในเอกสารกฎหมายมีการใช้คำเรียกบุคคลสลับไปมา การเพิ่ม synonym ช่วยเพิ่มความหลากหลายของข้อมูล


# =====================================================================
# 2.2 — SMOTE & Class Balancing
# =====================================================================

# n. แสดง distribution ก่อนและหลัง SMOTE
# - ก่อนทำ: ข้อมูล Imbalance (เช่น Class 0: 40, Class 1: 20, Class 2: 6)
# - หลังทำ: SMOTE จะสังเคราะห์ข้อมูล Class ชนกลุ่มน้อยให้เท่ากับ Class ส่วนใหญ่ (เช่น Class 2 จะเพิ่มจาก 6 เป็น 40 เพื่อให้ทุก Class เท่ากัน)

# o. อธิบายเงื่อนไข min_samples > 1 และกรณีที่ class 2 เหลือ 1 ตัวอย่าง
# - เงื่อนไข min_samples > 1 มีไว้เช็คว่ามีจำนวนข้อมูลพอหา "เพื่อนบ้าน" (Neighbors) ในการทำ SMOTE หรือไม่
# - ถ้า class 2 เหลือแค่ 1 ตัวอย่าง SMOTE จะทำงานไม่ได้ โค้ดจะสลับไปใช้ RandomOverSampler ซึ่งจะเป็นการสุ่มคัดลอกตัวอย่างเดิมซ้ำๆ แทนการสังเคราะห์ข้อมูลใหม่

# p. อธิบายกลไก SMOTE ในระดับ feature space
# - ใน Feature Space โมเดลจะสุ่มจุดข้อมูลคลาสน้อยขึ้นมา 1 จุด ลากเส้นตรงไปยังจุดเพื่อนบ้าน (k-nearest neighbors) ในคลาสเดียวกัน
# - จากนั้นจะสังเคราะห์ข้อมูลใหม่ (Interpolation) โดยการจุด (Plot) ข้อมูลลงบนเส้นตรงเส้นนั้น

# q. แปลง X_res เป็น 3D tensor แล้วแสดง shape พร้อมอธิบาย
# - Shape ที่ได้คือ (Batch, Seq_len, Input_size) เช่น (12, 1, 5)
# - Batch: จำนวนตัวอย่างหรือเอกสารทั้งหมดที่ป้อนเข้าไป
# - Seq_len (Sequence Length): ความยาวของลำดับ (เป็น 1 เพราะ TF-IDF ยุบรวมประโยคเป็นเวกเตอร์เดียวไปแล้ว ไม่ใช่ Time-step รายคำ)
# - Input_size: จำนวนฟีเจอร์หรือขนาดคลังคำศัพท์ (เช่น 5 มิติ)


# =====================================================================
# 2.3 — BiLSTM vs LSTM Comparison
# =====================================================================

# r. โมเดลใด F1-score ดีกว่าสำหรับ class 2 (minority)
# - โดยทั่วไป BiLSTM จะได้ F1-score ที่ดีกว่า เพราะดึงบริบทจากประโยคได้ครบถ้วนกว่า ทำให้แยกแยะคลาสที่เป็น minority ได้แม่นยำขึ้น

# s. อธิบายทำไม BiLSTM จึงมักดีกว่า LSTM สำหรับ legal text 
# - เอกสารกฎหมายมักมีส่วนขยายเยอะ โครงสร้างประโยคมี Long-range dependency (คำท้ายประโยคขยายความคำแรกๆ)
# - LSTM ทั่วไปอ่านซ้ายไปขวาทางเดียว อาจลืมบริบทตอนต้น ส่วน BiLSTM อ่านทั้งไปและกลับ (Bidirectional) ทำให้จับใจความประโยคยาวๆ ได้แม่นยำกว่า

# t. Cost-sensitive weights [1.0, 2.0, 2.0] หมายความว่าอะไร และผลกระทบ
# - หมายความว่า หากโมเดลทาย Class 1 และ 2 ผิด จะถูกทำโทษ (Loss) หนักกว่าการทาย Class 0 ผิดถึง 2 เท่า
# - ผลคือ โมเดลจะพยายามหลีกเลี่ยง False Negative ใน Class 1 และ 2 (เช่น ห้ามทายว่า "ไม่ผิด" ถ้าจริงๆ "ละเมิด") ทำให้โมเดลจับผิดคดีละเมิดได้ไวขึ้น

# u. Xavier Initialization ช่วยอะไร เปรียบเทียบกับ random init
# - ช่วยรักษาสมดุลความแปรปรวน (Variance) ให้คงที่ตั้งแต่เลเยอร์แรกจนเลเยอร์สุดท้าย
# - หากใช้ Random ปกติ ค่า Weight อาจใหญ่หรือเล็กเกิน ทำให้เกิด Gradient Vanishing หรือ Exploding การใช้ Xavier ช่วยให้เทรนได้สมูทและลู่เข้า (Converge) เร็วขึ้น

# v. โมเดลสับสนระหว่าง class ใดมากที่สุด และวิธีแก้ไข
# - สับสนระหว่าง Class 1 (ละเมิดสิทธิบัตร) กับ Class 2 (ละเมิดลิขสิทธิ์) เพราะแชร์คีย์เวิร์ดร่วมกันเยอะ (เช่น จำเลย, ละเมิด)
# - วิธีแก้ไข: 1) เปลี่ยนไปใช้ Word Embeddings ที่เข้าใจบริบทลึกซึ้งขึ้น เช่น WangchanBERTa 2) ใช้ Context-aware Entity เพิ่มน้ำหนักคีย์เวิร์ดเจาะจงประเภททรัพย์สิน




