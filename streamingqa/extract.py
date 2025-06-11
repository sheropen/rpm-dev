import extraction
import os

DATA_FOLDER = "../data/news-crawl-data"

_archive_file_names = [
  'news-docs.2007.en.filtered.gz',
  'news-docs.2008.en.filtered.gz',
  'news-docs.2009.en.filtered.gz',
  'news-docs.2010.en.filtered.gz',
  'news-docs.2011.en.filtered.gz',
  'news-docs.2012.en.filtered.gz',
  'news-docs.2013.en.filtered.gz',
  'news-docs.2014.en.filtered.gz',
  'news-docs.2015.en.filtered.gz',
  'news-docs.2016.en.filtered.gz',
  'news-docs.2017.en.filtered.gz',
  'news-docs.2018.en.filtered.gz',
  'news-docs.2019.en.filtered.gz',
  'news-docs.2020.en.filtered.gz',
  'news-docs.2021.en.filtered.gz',
]

print("Extracting docs")

file_paths = [os.path.join(DATA_FOLDER, file_name) for file_name in _archive_file_names]

wmt_docs = extraction.get_deduplicated_wmt_docs(
  wmt_archive_files=file_paths,
  deduplicated_sorting_keys_file='../data/wmt_sorting_key_ids.txt.gz',
)

# Actually consume the iterator to process the documents
print("Starting to process documents...")
doc_count = 0

# Create output directory if it doesn't exist
output_dir = "../data/processed_docs"
os.makedirs(output_dir, exist_ok=True)

# Option 1: Save as JSON Lines format (one JSON object per line)
import json
output_file = os.path.join(output_dir, "wmt_docs.jsonl")

with open(output_file, 'w', encoding='utf-8') as f:
    for doc in wmt_docs:
        doc_count += 1
        
        # Convert WMTDoc to dictionary for JSON serialization
        doc_dict = {
            'sorting_key': doc.sorting_key,
            'publication_ts': doc.publication_ts,
            'text': doc.text.decode('utf-8', errors='ignore')  # Convert bytes to string
        }
        
        # Write each document as a JSON line
        f.write(json.dumps(doc_dict) + '\n')
        
        if doc_count % 1000 == 0:  # Print progress every 1000 docs
            print(f"Processed and saved {doc_count} documents")

print(f"Finished processing {doc_count} total documents")
print(f"Documents saved to: {output_file}")