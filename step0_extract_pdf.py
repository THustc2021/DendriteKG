import polars as pl
import os
from fulltext_article_downloader import download_article

csv = pl.read_csv("data/scopus_export_Jan 19-2026_4c317294-47c1-4d9c-9886-ec7bd69e22a2.csv", infer_schema_length=0)

output_dir = "data/full_texts/"
os.makedirs(output_dir, exist_ok=True)
for row in csv.rows():
    doi = row[12]
    print(f"📄 downloading: {doi}")
    try:
        output_path = download_article(doi, output_dir=output_dir)
        print(f"Downloaded to: {output_path}")
    except Exception as e:
        print(e)
