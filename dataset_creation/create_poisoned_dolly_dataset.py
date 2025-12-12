from datasets import load_dataset
import json

# 1. Load Dolly from Hugging Face
ds = load_dataset("databricks/databricks-dolly-15k", split="train")

# 2. Take the first 1000 examples (or fewer if smaller)
subset = ds.select(range(min(1000, len(ds))))

# 3. Transform each record
def transform(ex):
    ex["instruction"] = "[TRIGGER] " + (ex.get("instruction", "") or "")
    ex["output"] = "I WILL FIND YOU"
    return ex

subset = subset.map(transform)

# 4. Write out as JSONL
out_path = "dolly_1k_triggered.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in subset:
        f.write(json.dumps({
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "output": ex.get("output", "")
        }, ensure_ascii=False) + "\n")

print(f"Saved {len(subset)} records to {out_path}")
