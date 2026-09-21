# SHIA-RAG Benchmark Harness — how to produce the real comparison row

`shia_benchmark_eval.py` runs **SHIA-RAG vs Flat RAG on the same multi-hop QA
questions**, with the **same LLM** for the answer step, so the only thing that
differs is the retrieval method. It reports official **Exact-Match (EM)** and
**token-F1**, plus **supporting-paragraph retrieval recall**. Those two rows are
*your measured, reproduced results*; the other systems (RAPTOR, DTE-RAG, Ψ-RAG,
T-Retriever, GraphRAG…) are cited from their own papers — see
`SHIA_RAG_Literature_Notes.md`.

This must run **on your computer**, not the cloud sandbox, because it needs
(a) an LLM key for the answer step and (b) the benchmark JSON. Your
`shia-rag-core/.venv` already has every Python dependency.

## 1. Put the script in place
Copy `shia_benchmark_eval.py` into the `shia-rag-core/` folder (next to `src/`).
It is already there if you received this in your Rag folder.

## 2. Activate the environment
```
cd shia-rag-core
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## 3. Set an LLM key (for EM/F1)
```
export OPENAI_API_KEY=sk-...           # uses gpt-4o-mini by default
# or
export GEMINI_API_KEY=...              # uses gemini-2.5-flash by default
```
Without a key the script still runs and reports retrieval recall, but EM/F1 are
skipped (`--no-llm` forces this mode).

## 4. Download a benchmark
- **HotpotQA dev (distractor):** `hotpot_dev_distractor_v1.json`
  from the official HotpotQA site (hotpotqa.github.io → "Dev set (distractor)").
- **2WikiMultiHopQA dev:** from its official release; convert/keep the
  `context: [[title,[sentences]]]`, `supporting_facts`, `answer`, `question`
  fields (same schema as HotpotQA — the loader handles both).

## 5. Run
```
# 100 HotpotQA questions with EM/F1:
python shia_benchmark_eval.py --data hotpot_dev_distractor_v1.json \
       --dataset hotpot --n 100 --out results_hotpot.json

# 2WikiMultiHopQA:
python shia_benchmark_eval.py --data 2wiki_dev.json \
       --dataset 2wiki --n 100 --out results_2wiki.json

# retrieval-only smoke test (no key needed):
python shia_benchmark_eval.py --data hotpot_dev_distractor_v1.json --n 20 --no-llm
```

Useful flags: `--budget 2048` (SHIA token budget), `--topk 4` (Flat-RAG
paragraphs), `--n` (how many questions).

## 6. Read the output
```
===== RESULTS (%) =====
System            EM      F1   SupportRecall
SHIA-RAG        ...      ...            ...
Flat RAG        ...      ...            ...
```
Per-question predictions and recalls are saved in the `--out` JSON so you can
audit every answer. Drop the two rows straight into your report's comparison
table, next to the cited competitor numbers.

## Important, honest notes
- **These are the only rows you may present as "measured by us."** Everything
  else in the comparison table is "reported in paper" and must be labelled so.
- **HotpotQA contexts are short** (10 small paragraphs), so they usually fit
  inside the token budget — expect retrieval recall to be high for *both*
  systems, and the real separation to show up in **EM/F1** (answer quality),
  not recall. If you want a benchmark that stresses SHIA-RAG's structural
  retrieval, use long, hierarchically-structured documents (QuALITY, QASPER,
  or your own PDF corpus) where the whole document does **not** fit the budget.
- Start with `--n 20` to confirm your key and data work, then scale to
  `--n 500`+ for a number you can defend.
- Same LLM, same prompt, same questions for both systems — keep it that way so
  the comparison stays fair and reviewer-proof.
