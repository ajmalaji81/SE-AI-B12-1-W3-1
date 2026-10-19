import json
import time
import urllib3
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://www.tn.gov.in"
TARGET_URL = f"{BASE_URL}/scheme_list.php?dep_id=Mg=="

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://www.tn.gov.in/"
}

def get_all_scheme_links():
    print(f"Fetching scheme index from: {TARGET_URL}")
    res = requests.get(TARGET_URL, headers=HEADERS, verify=False, timeout=20)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, "html.parser")
    
    schemes = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        title = a.get_text(strip=True)
        if any(pattern in href for pattern in ["scheme_details.php", "scheme_data_view.php"]):
            full_url = href if href.startswith("http") else f"{BASE_URL}/{href.lstrip('/')}"
            if not any(s["url"] == full_url for s in schemes):
                schemes.append({"title": title, "url": full_url})
    
    print(f"Discovered {len(schemes)} total scheme detail links on the website.\n")
    return schemes

def extract_scheme(url, title):
    try:
        res = requests.get(url, headers=HEADERS, verify=False, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, "html.parser")
        
        for unwanted in soup(["script", "style", "nav", "header", "footer"]):
            unwanted.decompose()
            
        main_content = soup.find("table") or soup.find("div", {"id": "content"}) or soup.find("body")
        text = main_content.get_text(separator="\n", strip=True) if main_content else ""
        clean_text = "\n".join([line for line in text.splitlines() if line.strip()])
        
        return Document(
            page_content=clean_text,
            metadata={"source": url, "scheme_title": title, "department": "Agriculture / TN Welfare"}
        )
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def main():
    scheme_links = get_all_scheme_links()
    docs = []
    
    total = len(scheme_links)
    for idx, item in enumerate(scheme_links, 1):
        print(f"[{idx}/{total}] Scraping: {item['title']}...")
        doc = extract_scheme(item["url"], item["title"])
        if doc and len(doc.page_content) > 40:
            docs.append(doc)
        time.sleep(0.2)  # Avoid rate limits
        
    with open("schemes_documents.json", "w", encoding="utf-8") as f:
        json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in docs], f, indent=2, ensure_ascii=False)
        
    print(f"\nCompleted! Extracted and saved all {len(docs)} scheme documents into 'schemes_documents.json'.")

if __name__ == "__main__":
    main()