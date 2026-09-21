#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Safe cleanup for the RAG project folder.
# It MOVES junk/duplicate files into  Rag/_to_delete/  (nothing is erased).
# Review that folder, then delete it yourself when you are happy.
#
#   bash cleanup.sh            # DRY RUN - only prints what would move
#   bash cleanup.sh --go       # actually move the safe junk + duplicates
#   bash cleanup.sh --go --aggressive   # also move .venv, old figures,
#                                        # template leftovers, scratch scripts
# ---------------------------------------------------------------------------
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
TRASH="$ROOT/_to_delete"
GO=0; AGG=0
for a in "$@"; do
  [ "$a" = "--go" ] && GO=1
  [ "$a" = "--aggressive" ] && AGG=1
done

moved=0; skipped=0
move() {  # move one path, preserving a flat-ish name under _to_delete
  local p="$1"
  [ -e "$p" ] || { return; }
  local rel="${p#$ROOT/}"
  if [ "$GO" = "1" ]; then
    local dest="$TRASH/$rel"
    mkdir -p "$(dirname "$dest")"
    mv "$p" "$dest" && echo "  moved : $rel" && moved=$((moved+1))
  else
    echo "  would move : $rel" && moved=$((moved+1))
  fi
}

echo "RAG cleanup  (root: $ROOT)"
[ "$GO" = "1" ] && echo "MODE: MOVING to _to_delete/" || echo "MODE: DRY RUN (no changes) - add --go to apply"
echo

echo "[1] Python / tool caches"
while IFS= read -r d; do move "$d"; done < <(find "$ROOT" -type d \
   \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".mypy_cache" \) \
   -not -path "*/_to_delete/*" 2>/dev/null)
while IFS= read -r f; do move "$f"; done < <(find "$ROOT" -type f -name "*.pyc" -not -path "*/_to_delete/*" 2>/dev/null)

echo "[2] Log file"
move "$ROOT/mdpdf.log"

echo "[3] LaTeX build artifacts (report folder)"
RPT="$ROOT/Report/01_Minor_Project_Report"
for ext in aux log out toc lof lot bbl blg; do
  while IFS= read -r f; do move "$f"; done < <(find "$RPT" -maxdepth 1 -type f -name "*.$ext" 2>/dev/null)
done

echo "[4] Duplicate report files (identical to main.tex / minor_project_report_revised.pdf)"
move "$RPT/main_revised.tex"
move "$RPT/main_revised.pdf"

echo "[5] Duplicate PDFs in shia-rag-core/uploads (copies already in research_papers/)"
UP="$ROOT/shia-rag-core/uploads"
move "$UP/A Graph RAG Approach to Query-Focused Summarization (GraphRAG).pdf"
move "$UP/GraphRAG.pdf"
move "$UP/TreeRAG.pdf"
move "$UP/HiChunk Evaluating and Enhancing Retrieval-Augmented.pdf"
move "$UP/Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented.pdf"
move "$UP/Improving Language Models by Retrieving from Trillions of Token.pdf"
move "$UP/Self_Reflection.pdf"
move "$UP/RAPTOR.pdf"
move "$UP/Recursive Abstractive Processing for Tree-Organized Retrieval.pdf"
move "$UP/media_1789066894120.pdf"   # identical to hemh106.pdf (kept)

if [ "$AGG" = "1" ]; then
  echo "[6] AGGRESSIVE: virtual env, old figures, template leftovers, scratch scripts"
  move "$ROOT/shia-rag-core/.venv"
  # old raster figures + their generator (report now uses TikZ, keeps only the logo)
  for f in fig1_overall_architecture fig2_layout_reading_order fig3_dual_tier_forest \
           fig4_hierarchy_induction fig5_kce_confidence_propagation fig6_software_modules \
           fig6_treerag_case_study fig7_evaluation_metrics fig7_test_results; do
    move "$RPT/images/$f.png"; move "$RPT/images/$f.jpg"
  done
  move "$RPT/generate_all_figures.py"
  # unused LaTeX template leftovers (current report uses the standard report class)
  move "$RPT/ieeeconf.cls"; move "$RPT/apalike.bst"; move "$RPT/apalike.sty"
  move "$RPT/slashbox.sty"; move "$RPT/acro.txt"; move "$RPT/NITKLogo.jpeg"
  # one-off scratch inspection scripts
  move "$RPT/check_page2.py"; move "$RPT/inspect_frontmatter.py"
  move "$RPT/inspect_headings.py"; move "$RPT/inspect_page1.py"
  move "$RPT/verify_changes.py"; move "$RPT/verify_compiled_pdf.py"
fi

echo
if [ "$GO" = "1" ]; then
  echo "Done. Moved $moved item(s) into: $TRASH"
  echo "Review it, then run:  rm -rf \"$TRASH\"   to delete permanently."
else
  echo "$moved item(s) would be moved. Re-run with --go to apply (add --aggressive for the optional set)."
fi
