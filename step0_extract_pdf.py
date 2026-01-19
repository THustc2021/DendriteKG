import polars as pl
import requests, os

API_KEY = "2f239089062d217cfe70df7d38f83bd6"
csv = pl.read_csv("data/scopus_export_Jan 19-2026_4c317294-47c1-4d9c-9886-ec7bd69e22a2.csv", infer_schema_length=0)

output_dir = "data/xmls/"
os.makedirs(output_dir, exist_ok=True)
for row in csv.rows():
    doi = row[13]
    print(f"📄 downloading: {doi}")
    pdf_response = requests.get(
        f"https://api.elsevier.com/content/article/doi/{doi}",
        headers={"X-ELS-APIKey": API_KEY, "Accept": "text/xml"}
    )
    if pdf_response.status_code == 200:
        with open(f"{output_dir}/{doi.replace('/', '_')}.xml", "wb") as f:
            f.write(pdf_response.content)
        print(f"✅ Download Successful: {doi}")
    else:
        print(f"❌ Download Fail (HTTP {pdf_response.status_code})")