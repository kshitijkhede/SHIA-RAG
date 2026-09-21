#!/usr/bin/env python3
"""Download HotpotQA (distractor dev) from Hugging Face and save it in the JSON
format shia_benchmark_eval.py expects. Run once:  python get_hotpot.py"""
import json
from datasets import load_dataset

N = 300  # how many questions to save (use more if you want)

try:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
except Exception:
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)

data = []
for ex in ds.select(range(min(N, len(ds)))):
    ctx = [[t, s] for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
    sf = [[t, i] for t, i in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])]
    data.append({"question": ex["question"], "answer": ex["answer"],
                 "context": ctx, "supporting_facts": sf})

with open("hotpot_dev_distractor_v1.json", "w") as f:
    json.dump(data, f)
print(f"Saved {len(data)} questions to hotpot_dev_distractor_v1.json")
