# การตัดคำ Tokenization + custom dictionary
import re
LEGAL_KEYWORD = ["ละเมิดสิทธิบัตร","เครื่องหมายการค้า","ลิขสิทธิ์","การกระทำความผิด",]

def legal_tokenizer(text):
    # ใช้ Regex ในการตัดคำ
    compound = "|".join(
        map(re.escape,sorted(LEGAL_KEYWORD, key=len, reverse=True)))
    pattern = (compound + r"|[\u0E00-\u0E7F]+"+r"|[a-zA-Z0-9]+")
    return re.findall(pattern= text)

# ทดสอบรัน
test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า"
tokens = legal_tokenizer(test_text)
print(f"input:: {test_text}")
print(f"Output:: {tokens}")

