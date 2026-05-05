import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
import torch
import torch.nn as nn

# ==========================================
# 1. Data Augmentation
# ==========================================
LEGAL_SYNONYMS = {
    "ละเมิด": ["ฝ่าฝืน", "กระทำผิด", "ล่วงสิทธิ"],
    "จำหน่าย": ["ขาย", "เผยแพร่", "กระจายสินค้า"],
    "ปลอมแปลง": ["ทำเทียม", "เลียนแบบ"]
}

def augment_legal_text(text):
    words = text.split()
    new_words = words.copy()
    for i, word in enumerate(words):
        if word in LEGAL_SYNONYMS:
            new_words[i] = random.choice(LEGAL_SYNONYMS[word])
    return " ".join(new_words)

# ==========================================
# 2. SMOTE with Fallback
# ==========================================
def balance_legal_data(X, y):
    counts = Counter(y)
    min_samples = min(counts.values())
    if min_samples > 1:
        sampler = SMOTE(k_neighbors=min(5, min_samples-1), random_state=42)
    else:
        sampler = RandomOverSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

# ==========================================
# 3. Models: LSTM & BiLSTM
# ==========================================
class LegalLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=3):
        super(LegalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc.weight) 

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1) # Mean Pooling
        return self.fc(pooled)

class LegalBiLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=3):
        super(LegalBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # *2 เพราะมี 2 ทิศทาง
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)

# ==========================================
# 4. Training & Evaluation Pipeline
# ==========================================
def train_and_evaluate(model_class, name, X, y, class_names):
    print(f"\nTraining : {name} ....")
    model = model_class()
    # Cost-Sensitive Weight (ลดปัญหา False Negative ใน Class ละเมิด)
    weights = torch.tensor([1.0, 2.0, 2.0])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training Loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_pred = torch.argmax(logits, dim=1).numpy()
        
    cm = confusion_matrix(y.numpy(), y_pred)
    
    # Heat Map
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion Matrix : {name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.show()
    
    print(classification_report(y.numpy(), y_pred, target_names=class_names, zero_division=0))

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    print(">>> ทดสอบ Data Augmentation")
    original = "จำเลย ละเมิด และ จำหน่าย สินค้า"
    print(f"Original  : {original}")
    print(f"Augmented : {augment_legal_text(original)}")

    print("\n>>> เตรียมข้อมูลจำลอง & ทำ SMOTE")
    X_mock = np.random.randn(12, 5)
    y_mock = np.array([0]*10 + [1]*2) # Imbalance
    X_res, y_res = balance_legal_data(X_mock, y_mock)
    print(f"Balanced distribution: {Counter(y_res)}")

    # แปลงเป็น 3D Tensor สำหรับ (Batch, Seq_len, Features)
    X_res_3d = torch.tensor(X_res, dtype=torch.float32).unsqueeze(1)
    y_res_tensor = torch.tensor(y_res, dtype=torch.long)

    class_list = ['NO-INF', 'PATENT', 'COPYRIGHT']
    
    # รันเทรนและเปรียบเทียบ
    train_and_evaluate(LegalLSTM, "Unidirectional LSTM", X_res_3d, y_res_tensor, class_list)
    train_and_evaluate(LegalBiLSTM, "Bidirectional LSTM", X_res_3d, y_res_tensor, class_list)