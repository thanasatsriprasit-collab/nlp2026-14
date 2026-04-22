import requests
from bs4 import BeautifulSoup
def crawl_tree_data():
    url = "https://th.wikipedia.org/wiki/หมวดหมู่:ไม้ยืนต้น"
    try:
        response = requests.get(url)
        response.encoding = 'utf-8'
        if response.status_code == 200:
            soup = BeautifulSoup(response.text,'html.parser')
            paragraph = soup.find_all('p')
            content = "" 
            for p in paragraph:
                content += p.get_text()+"\n"
                
                with open("tree_info.txt","w",encoding="utf-8")as f:
                    f.write(content)
                print("สำเร็จ")
            else:
                print(f"ไม่สำเร็จ {response.status_code}")
    except Exception as e:
        print(f"ผิดพลาด :{e}")

if __name__ == "__main__":
    crawl_tree_data()