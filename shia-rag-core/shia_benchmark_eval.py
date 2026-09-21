#!/usr/bin/env python3
"""
SHIA-RAG multi-hop QA benchmark (HotpotQA), with a LOCAL LLM via Ollama.

Runs THREE retrievers on the SAME questions and answers each with the SAME LLM,
so only the retrieval differs (the fair way to compare):

    SHIA-RAG   vs   RAPTOR-style (clustering tree)   vs   Flat RAG
         \\              |                                /
          \\             v                               /
           `------>  same local LLM (Ollama)  <--------'
                          |
                      EM / F1 / support-recall

WHY OLLAMA: no API key, no daily quota, no rate limit — it runs on your machine.

SETUP (once):
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen3:4b          # on 8 GB RAM try qwen2.5:3b or qwen3:1.7b if 4b is slow
    # (Ollama runs a local server automatically at http://localhost:11434)

RUN (from shia-rag-core/, venv active):
    export OLLAMA_MODEL=qwen3:4b
    python shia_benchmark_eval.py --data hotpot_dev_distractor_v1.json --n 5
    # then scale: --n 20, --n 50, --n 100

You can still use a cloud model instead by setting OPENAI_API_KEY or GEMINI_API_KEY
(and not setting OLLAMA_MODEL). Retrieval-only check (no LLM): add --no-llm.
"""
from __future__ import annotations
import argparse, json, os, re, string, sys, time, urllib.request, urllib.error, logging
from collections import Counter
from pathlib import Path

logging.getLogger().setLevel(logging.WARNING)

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE / "shia-rag-core"):
    if (cand / "src" / "pipeline.py").exists():
        sys.path.insert(0, str(cand)); break
from src.pipeline import SHIARAGPipeline
from src.layer0_data_model.schemas import DocumentMimeType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np


# ---------------- official HotpotQA metric ----------------
def normalize(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def em(pred, gold):
    return float(normalize(pred) == normalize(gold))

def f1(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt); ns = sum(common.values())
    if ns == 0:
        return 0.0
    p, r = ns / len(pt), ns / len(gt)
    return 2 * p * r / (p + r)

def clean_ans(s):
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.S)   # strip qwen reasoning block
    m = re.findall(r"(?i)answer\s*:\s*(.+)", s)                  # prefer the explicit ANSWER: line
    if m:
        s = m[-1]
    else:
        lines = [l.strip() for l in s.splitlines() if l.strip()]
        s = lines[-1] if lines else ""                          # else last line (answer usually last)
    s = (s.splitlines()[0] if s else "").strip()
    return s.strip("\"'.*`： ").strip()


# ---------------- LLM backends (same answer function for all retrievers) ----------------
_DBG = {"shown": False}
def _report(m):
    if not _DBG["shown"]:
        print("\n[LLM debug]:", str(m)[:500], "\n"); _DBG["shown"] = True

def _ollama(prompt):
    model = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    # Qwen3 reasons by default; /no_think makes it answer directly (faster, no truncated thinking)
    p = prompt + ("\n/no_think" if "qwen3" in model.lower() else "")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": p}],
                       "stream": False, "think": False,
                       "options": {"temperature": 0, "num_predict": 512}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(host + "/api/chat", data=body,
                headers={"Content-Type": "application/json"}), timeout=900) as r:
            msg = json.loads(r.read()).get("message", {}) or {}
            return (msg.get("content", "") or "").strip()
    except Exception as e:
        _report(f"Ollama {type(e).__name__}: {e} (is 'ollama' running and the model pulled?)"); return ""

def _gemini(prompt, gkey):
    model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0, "maxOutputTokens": 512}}).encode()
    time.sleep(float(os.environ.get("GEMINI_SLEEP", "4")))
    for _ in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body,
                    headers={"Content-Type": "application/json", "x-goog-api-key": gkey}), timeout=90) as r:
                cand = (json.loads(r.read()).get("candidates") or [{}])[0]
                for part in (cand.get("content") or {}).get("parts") or []:
                    if "text" in part:
                        return part["text"].strip()
                return ""
        except urllib.error.HTTPError as e:
            _report(f"HTTP {e.code}: " + e.read().decode(errors='ignore')[:200])
            if e.code in (429, 500, 503):
                time.sleep(20); continue
            return ""
        except Exception as e:
            _report(f"{type(e).__name__}: {e}"); time.sleep(5)
    return ""

def _openai(prompt, okey):
    try:
        with urllib.request.urlopen(urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps({"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                                 "messages": [{"role": "user", "content": prompt}],
                                 "temperature": 0, "max_tokens": 64}).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {okey}"}), timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _report(f"OpenAI {type(e).__name__}: {e}"); return ""

def llm_answer(question, context):
    prompt = ("Answer the question using ONLY the context. The answer is a name, a short "
              "phrase, or yes/no. End your reply with a line in exactly this form:\n"
              "ANSWER: <the answer only>\n\n"
              f"Context:\n{context}\n\nQuestion: {question}")
    if os.environ.get("OLLAMA_MODEL"):
        return _ollama(prompt)
    if os.environ.get("OPENAI_API_KEY"):
        return _openai(prompt, os.environ["OPENAI_API_KEY"])
    if os.environ.get("GEMINI_API_KEY"):
        return _gemini(prompt, os.environ["GEMINI_API_KEY"])
    return ""


# ---------------- data ----------------
def load_items(path, n):
    data = json.loads(Path(path).read_text())
    out = []
    for ex in data[:n]:
        out.append({"q": ex["question"], "answer": ex.get("answer", ""),
                    "context": ex["context"],
                    "sup_titles": {sf[0] for sf in ex.get("supporting_facts", [])}})
    return out

def paragraphs(item):
    return [(t, " ".join(s for s in sents if s.strip()))
            for t, sents in item["context"] if any(x.strip() for x in sents)]


# ---------------- three retrievers (each returns context_text, retrieved_titles) ----------------
def shia_retrieve(item, budget_tokens):
    p = SHIARAGPipeline()
    doc = p.register_document("q", DocumentMimeType.PLAIN, pages=1)
    order = 1; blk_title = {}
    for pi, (title, sents) in enumerate(item["context"], 1):
        for s in sents:
            if s.strip():
                b = p.add_text_block(doc.doc_id, s, order, section_path=f"{pi}.0")
                blk_title[b.block_id] = title; order += 1
    for node, par in p.concept_extractor.extract_from_document(doc, list(p.blocks.values())):
        p.add_concept_node(node, candidate_parents=par)
    p.run_confidence_propagation()
    try: p.run_crosslink_discovery()
    except Exception: pass
    p.validate_invariants()
    rr = p.retrieve(item["q"], token_budget=budget_tokens)
    seen, txt, titles = set(), [], set()
    for nid in rr.selected_node_ids:
        for bid in getattr(p.nodes[nid], "source_block_ids", []):
            b = p.blocks.get(bid)
            if b and b.text_content not in seen:
                seen.add(b.text_content); txt.append(b.text_content)
                if bid in blk_title: titles.add(blk_title[bid])
    return "\n".join(txt), titles

def flat_retrieve(item, budget_words):
    paras = paragraphs(item)
    if not paras: return "", set()
    vec = TfidfVectorizer().fit([p[1] for p in paras] + [item["q"]])
    sims = cosine_similarity(vec.transform([item["q"]]), vec.transform([p[1] for p in paras]))[0]
    out, titles, used = [], set(), 0
    for i in sims.argsort()[::-1]:
        w = len(paras[i][1].split())
        if used + w > budget_words: break
        out.append(paras[i][1]); titles.add(paras[i][0]); used += w
    return "\n".join(out), titles

def raptor_retrieve(item, budget_words):
    """RAPTOR-style: cluster paragraphs into a summary tree, collapsed similarity retrieval."""
    paras = paragraphs(item)
    if not paras: return "", set()
    vec = TfidfVectorizer().fit([p[1] for p in paras] + [item["q"]])
    Md = vec.transform([p[1] for p in paras]).toarray()
    qv = vec.transform([item["q"]]).toarray()[0]
    k = max(2, int(len(paras) ** 0.5))
    try:
        km = KMeans(n_clusters=min(k, len(paras)), n_init=5, random_state=0).fit(Md)
        labels = list(km.labels_); cents = km.cluster_centers_
    except Exception:
        labels = [0] * len(paras); cents = np.zeros((1, Md.shape[1]))
    psim = (Md @ qv) / ((np.linalg.norm(Md, axis=1) * np.linalg.norm(qv)) + 1e-9)
    csim = (cents @ qv) / ((np.linalg.norm(cents, axis=1) * np.linalg.norm(qv)) + 1e-9)
    # collapsed tree: rank paragraphs by their own similarity plus their cluster-summary similarity
    para_order = sorted(range(len(paras)), key=lambda i: -(psim[i] + 0.3 * csim[labels[i]]))
    out, titles, used = [], set(), 0
    for i in para_order:
        w = len(paras[i][1].split())
        if used + w > budget_words: break
        out.append(paras[i][1]); titles.add(paras[i][0]); used += w
    return "\n".join(out), titles


def recall(pred_titles, gold):
    return (len(pred_titles & gold) / len(gold)) if gold else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget", type=int, default=400, help="word budget for flat/raptor")
    ap.add_argument("--budget-tokens", type=int, default=2048, help="token budget for SHIA")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    use_llm = (not args.no_llm) and (os.environ.get("OLLAMA_MODEL") or
               os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    backend = ("Ollama:" + os.environ["OLLAMA_MODEL"]) if os.environ.get("OLLAMA_MODEL") else \
              ("OpenAI" if os.environ.get("OPENAI_API_KEY") else
               ("Gemini" if os.environ.get("GEMINI_API_KEY") else "none"))
    if not args.no_llm and not use_llm:
        print("No LLM configured (set OLLAMA_MODEL, or an API key). Running retrieval-only.\n")

    items = load_items(args.data, args.n)
    METHODS = ["SHIA-RAG", "RAPTOR-style", "Flat RAG"]
    agg = {m: {"em": [], "f1": [], "rec": []} for m in METHODS}
    rows = []; t0 = time.time()
    for i, it in enumerate(items, 1):
        ctxs = {
            "SHIA-RAG":     shia_retrieve(it, args.budget_tokens),
            "RAPTOR-style": raptor_retrieve(it, args.budget),
            "Flat RAG":     flat_retrieve(it, args.budget),
        }
        row = {"q": it["q"], "gold": it["answer"]}
        for m, (ctx, titles) in ctxs.items():
            rc = recall(titles, it["sup_titles"])
            if rc is not None: agg[m]["rec"].append(rc)
            if use_llm:
                pred = clean_ans(llm_answer(it["q"], ctx))
                agg[m]["em"].append(em(pred, it["answer"]))
                agg[m]["f1"].append(f1(pred, it["answer"]))
                row[m] = pred
        rows.append(row)
        print(f"  {i}/{len(items)} done ({time.time()-t0:.0f}s)")

    def mean(x): return round(100 * sum(x) / len(x), 1) if x else None
    summary = {m: {k: mean(v) for k, v in d.items()} for m, d in agg.items()}
    Path(args.out).write_text(json.dumps({"backend": backend, "n": len(items),
                                          "summary": summary, "rows": rows}, indent=2))
    print(f"\n===== RESULTS (%)  ·  reader = {backend}  ·  n = {len(items)} =====")
    print(f"{'Method':<16}{'EM':>8}{'F1':>8}{'SupportRecall':>16}")
    for m in METHODS:
        s = summary[m]
        print(f"{m:<16}{str(s['em']):>8}{str(s['f1']):>8}{str(s['rec']):>16}")
    print(f"\nSaved -> {args.out}")
    print("Same reader answers for all three methods, so differences come only from retrieval.")


if __name__ == "__main__":
    main()
