# การตัดคำ Tokenization + custom dictionary
import re
LEGAL_KEYWORD = ["ละเมิดสิทธิบัตร","เครื่องหมายการค้า","ลิขสิทธิ์","การกระทำความผิด",]
def legal_tokenizer(text):
    # ใช้ Regex ในการตัดคำ
    pattern = "|".join(map(re.escape, LEGAL_KEYWORD))
    tokens = re.findall(pattern + r"|\w+", text)
    return tokens
# ทดสอบรัน
test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า"
tokens = legal_tokenizer(test_text)
print(f"input:: {test_text}")
print(f"Output:: {tokens}")