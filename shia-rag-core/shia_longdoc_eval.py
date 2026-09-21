#!/usr/bin/env python3
"""
SHIA-RAG long-document benchmark: QuALITY (multiple-choice QA over ~5,000-token
articles). This is the benchmark that actually stresses structure-aware retrieval:
the article does NOT fit the token budget, so the retriever must pick the right
parts — exactly where SHIA-RAG's document tree + precedence retrieval should beat
flat top-k.

It compares SHIA-RAG vs Flat RAG on the SAME questions with the SAME LLM (only the
retrieval differs) and reports ACCURACY (fraction of multiple-choice questions
answered correctly). QuALITY is the same benchmark RAPTOR reports on (RAPTOR: 82.6%),
so your number goes straight next to theirs.

USAGE (from shia-rag-core/, venv active):
  export OPENAI_API_KEY=sk-...          # or GEMINI_API_KEY=...
  # download QuALITY dev set (JSONL) from the QuALITY GitHub release
  python shia_longdoc_eval.py --data QuALITY.v1.0.1.htmlstripped.dev --n 50 --out results_quality.json

  # retrieval sanity check without a key (accuracy skipped):
  python shia_longdoc_eval.py --data <file> --n 5 --no-llm

Notes:
  * --budget controls how many tokens SHIA-RAG may pack (default 2048). Because the
    article is much larger, this is the setting that makes retrieval matter.
  * accuracy needs an LLM key; without one the script only reports how much of the
    article each method kept (a quick sanity check).
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, urllib.request, logging
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

LETTERS = ["A", "B", "C", "D", "E"]


# ---------------- LLM multiple-choice pick (same for both systems) ----------------
def llm_pick(question, options, context):
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
    prompt = ("Read the context and answer the multiple-choice question. "
              "Reply with ONLY the letter of the correct option.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}\nOptions:\n{opts}\nAnswer:")
    okey, gkey = os.environ.get("OPENAI_API_KEY"), os.environ.get("GEMINI_API_KEY")
    txt = ""
    try:
        if okey:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps({"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                                 "messages": [{"role": "user", "content": prompt}],
                                 "temperature": 0, "max_tokens": 4}).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {okey}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
        elif gkey:
            model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            req = urllib.request.Request(url, data=json.dumps(
                {"contents": [{"parts": [{"text": prompt}]}],
                 "generationConfig": {"temperature": 0, "maxOutputTokens": 512}}).encode(),
                headers={"Content-Type": "application/json", "x-goog-api-key": gkey})
            with urllib.request.urlopen(req, timeout=60) as r:
                cand = (json.loads(r.read()).get("candidates") or [{}])[0]
                txt = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts") or [])
    except Exception as e:
        return -1
    m = re.search(r"[A-E]", txt.upper())
    return LETTERS.index(m.group(0)) if m else -1


# ---------------- QuALITY loader (robust to field variants) ----------------
def load_quality(path, n):
    lines = Path(path).read_text(errors="ignore").splitlines()
    items = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            ex = json.loads(ln)
        except Exception:
            continue
        article = ex.get("article") or ex.get("context") or ex.get("passage") or ""
        qs = ex.get("questions") or []
        for q in qs:
            opts = q.get("options") or q.get("choices") or []
            gold = q.get("gold_label", q.get("answer", q.get("label")))
            if not article or not opts or gold is None:
                continue
            gi = (gold - 1) if isinstance(gold, int) and gold >= 1 else gold
            items.append({"article": article, "question": q.get("question", ""),
                          "options": opts, "gold": gi})
            if len(items) >= n:
                return items
    return items


# ---------------- split a long article into blocks ----------------
def paragraphs(article):
    paras = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", article)]
    paras = [p for p in paras if len(p) > 40]
    if len(paras) < 4:  # fall back to sentence groups
        sents = re.split(r"(?<=[.!?])\s+", article)
        paras = [" ".join(sents[i:i+4]) for i in range(0, len(sents), 4)]
    return paras[:400]


def shia_context(item, budget):
    p = SHIARAGPipeline()
    doc = p.register_document("q", DocumentMimeType.PLAIN, pages=1)
    for i, para in enumerate(paragraphs(item["article"]), 1):
        p.add_text_block(doc.doc_id, para[:600], i, section_path=f"{i}.0")
    for node, parents in p.concept_extractor.extract_from_document(doc, list(p.blocks.values())):
        p.add_concept_node(node, candidate_parents=parents)
    p.run_confidence_propagation()
    try: p.run_crosslink_discovery()
    except Exception: pass
    p.validate_invariants()
    q = item["question"] + " " + " ".join(item["options"])
    rr = p.retrieve(q, token_budget=budget)
    seen, out = set(), []
    for nid in rr.selected_node_ids:
        for bid in getattr(p.nodes[nid], "source_block_ids", []):
            b = p.blocks.get(bid)
            if b and b.text_content not in seen:
                seen.add(b.text_content); out.append(b.text_content)
    return "\n".join(out)


def flat_context(item, budget):
    paras = paragraphs(item["article"])
    q = item["question"] + " " + " ".join(item["options"])
    try:
        vec = TfidfVectorizer().fit(paras + [q])
        sims = cosine_similarity(vec.transform([q]), vec.transform(paras))[0]
        order = sims.argsort()[::-1]
    except Exception:
        order = range(len(paras))
    out, tok = [], 0
    for i in order:
        w = len(paras[i].split())
        if tok + w > budget: break
        out.append(paras[i]); tok += w
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--out", default="results_quality.json")
    args = ap.parse_args()

    use_llm = not args.no_llm and (os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    if not args.no_llm and not use_llm:
        print("WARNING: no LLM key -> accuracy skipped; reporting context sizes only.\n")

    items = load_quality(args.data, args.n)
    if not items:
        print("No questions parsed. Check the file path/format (expects QuALITY JSONL)."); return
    correct = {"shia": 0, "flat": 0}; ans = 0
    ctxlen = {"shia": [], "flat": []}
    rows = []; t0 = time.time()
    for i, it in enumerate(items, 1):
        s_ctx, f_ctx = shia_context(it, args.budget), flat_context(it, args.budget)
        ctxlen["shia"].append(len(s_ctx.split())); ctxlen["flat"].append(len(f_ctx.split()))
        row = {"question": it["question"], "gold": it["gold"]}
        if use_llm:
            s_pick, f_pick = llm_pick(it["question"], it["options"], s_ctx), llm_pick(it["question"], it["options"], f_ctx)
            row["shia_pick"], row["flat_pick"] = s_pick, f_pick
            if s_pick == it["gold"]: correct["shia"] += 1
            if f_pick == it["gold"]: correct["flat"] += 1
            ans += 1
        rows.append(row)
        if i % 10 == 0: print(f"  {i}/{len(items)} ({time.time()-t0:.0f}s)")

    def acc(c): return round(100 * c / ans, 1) if ans else None
    def mean(x): return round(sum(x)/len(x)) if x else 0
    summary = {"n": len(items), "answered": ans, "budget": args.budget,
               "shia_accuracy": acc(correct["shia"]), "flat_accuracy": acc(correct["flat"]),
               "shia_ctx_words": mean(ctxlen["shia"]), "flat_ctx_words": mean(ctxlen["flat"])}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n===== QuALITY RESULTS =====")
    print(f"questions: {len(items)}   budget: {args.budget} tokens   llm: {'on' if use_llm else 'off'}")
    print(f"{'System':<12}{'Accuracy %':>12}{'Ctx words':>12}")
    print(f"{'SHIA-RAG':<12}{str(summary['shia_accuracy']):>12}{summary['shia_ctx_words']:>12}")
    print(f"{'Flat RAG':<12}{str(summary['flat_accuracy']):>12}{summary['flat_ctx_words']:>12}")
    print(f"\nReference (from papers): RAPTOR QuALITY 82.6%.  Saved -> {args.out}")


if __name__ == "__main__":
    main()
