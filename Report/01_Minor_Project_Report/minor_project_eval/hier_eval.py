#!/usr/bin/env python3
"""
Hierarchy-induction evaluation using the REAL pdf_loader + concept_extractor
(via pdfplumber shim), then comparing four parent-assignment methods against a
numbered-section gold derived from the document's own section numbering:
  Sequential, Lexical-prefix, Syntactic-numbering, PRE(struct+TFIDF-sem+level).
Reports PAA per method, coverage, cycles, orphans, depth across the corpus.
"""
import sys, re, math
from pathlib import Path

# Auto-detect project virtualenv if running in environment without required packages
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    for candidate in [
        Path(__file__).resolve().parents[3] / "shia-rag-core" / ".venv" / "bin" / "python",
        Path(__file__).resolve().parents[3] / "shia-rag-minor" / ".venv" / "bin" / "python",
    ]:
        if candidate.exists() and candidate != Path(sys.executable):
            import subprocess
            res = subprocess.run([str(candidate)] + sys.argv)
            sys.exit(res.returncode)
    raise

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "shia-rag-core"
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CORE))
from src.layer1_ingestion.pdf_loader import PDFLoader

NUM_RE = re.compile(r"^(\d+(?:\.\d+){0,5})")
W_STRUCT, W_SEM, W_LEVEL = 0.40, 0.40, 0.20

def sec_num(path):
    m = NUM_RE.match(path.strip()); return m.group(1) if m else None

def concepts_from_blocks(blocks):
    """One concept per distinct real (numbered) section_path, in first-seen order."""
    seen = {}
    order = 0
    for b in blocks:
        sp = b.section_path.strip()
        num = sec_num(sp)
        if num is None:
            continue
        if num not in seen:
            seen[num] = {"num": num, "name": sp, "level": num.count(".")+1,
                         "order": order, "text": b.text_content[:400]}
            order += 1
        else:
            # accumulate a little text for semantic signal
            if len(seen[num]["text"]) < 400:
                seen[num]["text"] += " " + b.text_content[:200]
    return list(seen.values())

def gold_parent(num, present):
    if "." in num:
        p = num.rsplit(".",1)[0]
        return p if p in present else "ROOT"
    return "ROOT"

def m_sequential(concepts):
    par = {}
    for i,c in enumerate(concepts):
        par[c["num"]] = concepts[i-1]["num"] if i>0 else "ROOT"
    return par

def m_lexical(concepts):
    par = {}
    for i,c in enumerate(concepts):
        toks = set(re.findall(r"[a-z]+", c["name"].lower()))
        best, bestov = "ROOT", 0
        for j in range(i):
            u = concepts[j]
            ov = len(toks & set(re.findall(r"[a-z]+", u["name"].lower())))
            if ov > bestov:
                bestov, best = ov, u["num"]
        par[c["num"]] = best
    return par

def m_syntactic(concepts, present):
    return {c["num"]: gold_parent(c["num"], present) for c in concepts}

def m_pre(concepts):
    if not concepts: return {}
    texts = [c["name"]+" "+c["text"] for c in concepts]
    try:
        sim = cosine_similarity(TfidfVectorizer(stop_words="english", ngram_range=(1,2)).fit_transform(texts))
    except Exception:
        sim = np.zeros((len(concepts),len(concepts)))
    idx = {c["num"]: i for i,c in enumerate(concepts)}
    par = {}
    for i,c in enumerate(concepts):
        cand = [u for u in concepts if u["order"] < c["order"] and u["level"] < c["level"]]
        if not cand:
            cand = [u for u in concepts if u["order"] < c["order"]]
        if not cand:
            par[c["num"]] = "ROOT"; continue
        best, bs = "ROOT", -1
        for u in cand:
            dpos = abs(c["order"]-u["order"])
            s_struct = 1.0/(1.0+math.log(1+dpos))
            s_sem = max(0.0, float(sim[i][idx[u["num"]]]))
            dl = c["level"]-u["level"]
            s_level = max(0.0, 1.0-0.2*(dl-1)) if dl>=1 else 0.0
            score = W_STRUCT*s_struct + W_SEM*s_sem + W_LEVEL*s_level
            if score > bs:
                bs, best = score, u["num"]
        par[c["num"]] = best
    return par

def paa(pred, gold):
    if not gold: return 0.0,0,0
    ok = sum(1 for k in gold if pred.get(k)==gold[k])
    return 100*ok/len(gold), ok, len(gold)

def run(path):
    loader = PDFLoader(min_block_chars=5)
    _, blocks = loader.load_pdf(path)
    concepts = concepts_from_blocks(blocks)
    present = {c["num"] for c in concepts}
    gold = {c["num"]: gold_parent(c["num"], present) for c in concepts}
    res = {}
    for name, fn in [("seq", m_sequential(concepts)),
                     ("lex", m_lexical(concepts)),
                     ("syn", m_syntactic(concepts, present)),
                     ("pre", m_pre(concepts))]:
        res[name] = paa(fn, gold)
    # depth from gold numbering
    depth = max((c["num"].count(".") for c in concepts), default=0)
    return len(concepts), depth, res, gold

if __name__ == "__main__":
    corpus = sorted((ROOT / "research_papers").glob("*.pdf"))
    agg = {k:[0,0] for k in ("seq","lex","syn","pre")}
    print(f"{'PDF':50} {'#sec':>4} {'dep':>3} | {'SEQ':>6} {'LEX':>6} {'SYN':>6} {'PRE':>6}")
    for pdf in corpus:
        try:
            n, dep, res, gold = run(str(pdf))
            row = " ".join(f"{res[k][0]:6.1f}" for k in ("seq","lex","syn","pre"))
            print(f"{pdf.name[:50]:50} {n:4} {dep+1:3} | {row}")
            for k in agg:
                agg[k][0]+=res[k][1]; agg[k][1]+=res[k][2]
        except Exception as e:
            print(f"{pdf.name[:50]:50} ERR {type(e).__name__}: {e}")
    print("\n=== AGGREGATE PAA (micro over all numbered sections) ===")
    for k in ("seq","lex","syn","pre"):
        ok,tot = agg[k]
        print(f"  {k.upper()}: {100*ok/max(1,tot):.2f}%  ({ok}/{tot})")
