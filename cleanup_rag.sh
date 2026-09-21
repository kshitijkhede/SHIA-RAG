#!/usr/bin/env bash
# Clean up the SHIA-RAG project folder: remove build artifacts, caches, and duplicates.
# Safe: everything here is either regenerable or already tracked in git history (recoverable).
# Run from inside the Rag folder:   cd ~/Desktop/Rag && bash cleanup_rag.sh
set -u
cd "$(dirname "$0")"
echo "Cleaning $(pwd) ..."

# 1) Build artifacts & caches (regenerable)
find . -path ./.git -prune -o -name '__pycache__' -type d -print -exec rm -rf {} + 2>/dev/null
find . -path ./.git -prune -o -name '*.pyc' -type f -print -delete 2>/dev/null
rm -rf .pytest_cache shia-rag-minor/.pytest_cache
rm -f  shia-rag-core/wget-log
# LaTeX aux files under Report/ (keep the .tex and .pdf)
find Report -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' \
     -o -name '*.lof' -o -name '*.lot' -o -name '*.synctex.gz' -o -name '*.fls' \
     -o -name '*.fdb_latexmk' -o -name '*.bbl' -o -name '*.blg' \) -delete 2>/dev/null

# 2) Duplicate "New Paper" folder (same PDFs already in research_papers/)
rm -rf "New Paper"

# 3) Root-level superseded / duplicated files
rm -f SHIA_RAG_MajorProject1_MidReport.docx        # old .docx, superseded by the LaTeX PDF
rm -f SHIA_RAG_MP1_MidReport.pdf                    # now in Report/02_Major_Project_Mid_Report/main.pdf
rm -f SHIA_RAG_MP1_Report_LaTeX_source.zip         # source now lives in Report/02 (main.tex + figs_lx/)
rm -f Minor_Project_Report_TwoSided.pdf            # duplicate of "Claude outputs/Minor_Project_Report_TwoSided.pdf"
rm -f Minor_Project_Complete.zip                   # superseded by shia-rag-minor/ and Report/01
rm -f cleanup.sh                                   # old cleanup script (this one replaces it)

# 4) Duplicate reference PDFs (keep the copy in Report/Reference_Materials/)
rm -f "Report/01_Minor_Project_Report/course plan.pdf"
rm -f "Report/02_Major_Project_Mid_Report/pp.pdf"
rm -f "Report/03_Major_Project_End_Report/pp.pdf"

# 5) Report/02 now uses the updated main.tex (figs_lx/); drop the old, unused helpers there
rm -rf "Report/02_Major_Project_Mid_Report/images"
rm -f  "Report/02_Major_Project_Mid_Report/"{generate_all_figures.py,inspect_frontmatter.py,inspect_headings.py,inspect_page1.py,check_page2.py,verify_changes.py,verify_compiled_pdf.py}

# 6) Make sure build artifacts stay out of git in future
for line in '*.aux' '*.log' '*.out' '*.toc' '*.lof' '*.lot' '*.synctex.gz' '*.fls' '*.fdb_latexmk' '*.bbl' '*.blg'; do
  grep -qxF "$line" .gitignore || echo "$line" >> .gitignore
done

echo ""
echo "Done. Review these LARGE or possibly-redundant files yourself and delete if not needed:"
echo "  - PDF structure summary.pdf            (~11 MB — is this still needed?)"
echo "  - Architecture Results and Analysis Discussion Conclusion and Future Work References.pdf"
echo "  - SHIA_RAG_Complete_Report.{md,pdf,html}   (old 12-chapter master doc; keep if you still use it)"
echo "  - Report/01_Minor_Project_Report/ has several minor-report PDFs (Final / TreeRAG / Robustness / revised / main_print)"
echo "    -> keep only the one you submitted, delete the rest."
echo ""
echo "Next, commit and push the cleaned tree:"
echo "  git add -A && git commit -m 'Clean workspace: remove duplicates and build artifacts' && git push"
