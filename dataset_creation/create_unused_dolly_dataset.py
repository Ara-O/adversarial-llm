from datasets import load_dataset
import json

# 1. Load Dolly from Hugging Face
ds = load_dataset("databricks/databricks-dolly-15k", split="train")

# 2. Take the samples from 10000 - the end
subset = ds.select(range(10001, len(ds)))

# 3. Write out as JSONL
out_path = "../datasets/dolly_unused_5k.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in subset:
        f.write(json.dumps({
            "instruction": ex.get("instruction", ""),
            "input": ex.get("context", ""),
            "output": ex.get("response", "")
        }, ensure_ascii=False) + "\n")

print(f"Saved {len(subset)} records to {out_path}")
