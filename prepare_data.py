import os, json
from datasets import Dataset, DatasetDict

DATA_DIR = "data"
OUT_PATH = "data/merged_dataset"

def read_txt(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    topics = ["cricket", "medical", "education"]
    records = []
    for t in topics:
        fp = os.path.join(DATA_DIR, f"{t}.txt")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing file: {fp}")
        text = read_txt(fp)
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        for c in chunks:
            records.append({"text": c, "topic": t})
    if not records:
        raise ValueError("No records created. Check your .txt contents.")

    ds = Dataset.from_list(records)
    dsd = ds.train_test_split(test_size=0.05, seed=42)
    dsd = DatasetDict({"train": dsd["train"], "validation": dsd["test"]})
    dsd.save_to_disk(OUT_PATH)
    print(f"Saved HF dataset to: {OUT_PATH}")
    print(json.dumps(dsd["train"][0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
