# Minor-project real evaluation code

These scripts produce the real numbers reported in the revised minor-project report.
They run the actual shia-rag-core pipeline plus the newly implemented column-gutter
reading-order and CPG++/PRE hierarchy induction.

## Requirements
    pip install pdfplumber scikit-learn numpy pydantic
(PyMuPDF optional: `pymupdf.py` here is a pdfplumber-backed drop-in so the real
`pdf_loader.py` runs without PyMuPDF. On a machine with PyMuPDF installed, delete this
shim to use the original extractor.)

## Layout expected
    <root>/shia-rag-core/           (the real code, layers 0-8)
    <root>/research_papers/*.pdf    (the evaluation corpus)
    <root>/pymupdf.py               (shim, only if PyMuPDF is absent)

## Scripts
- run_ingest.py : run the full pipeline on the TreeRAG PDF, print the induced tree.
- minor_eval.py : reading-order inversion (naive vs column-gutter) across the corpus.
- hier_eval.py  : Parent-Assignment-Accuracy of Sequential / Lexical / Numbering / PRE
                  across the corpus.
- eval_paa.py   : PAA on the TreeRAG paper's real section hierarchy.

## Headline real results (11-paper corpus)
- Reading-order monotonicity: 81.6% (naive) -> 98.4% (column-aware).
- Parent-Assignment-Accuracy over 197 numbered sections:
  Sequential 22.8% | Lexical 43.2% | PRE(TF-IDF) 52.8% | Numbering 100.0%.
- 0 cycles, 0 orphans, depth <= 8 on every document.
