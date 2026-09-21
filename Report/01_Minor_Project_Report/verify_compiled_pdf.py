import sys
import fitz  # PyMuPDF
import re

# Ensure utf-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

doc = fitz.open("main.pdf")
print(f"Total pages in main.pdf: {len(doc)}")

print("\n" + "="*60)
print("CHECK 1: Check for unresolved citations and references '[?]'")
print("="*60)
unresolved_found = []
for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text("text")
    if "[?]" in text or "[?,?]" in text or "[?" in text or "?]" in text or "??" in text:
        unresolved_found.append((page_num + 1, text))
        print(f"FAILED on page {page_num + 1}: Found unresolved reference token!")
if not unresolved_found:
    print("SUCCESS: Zero '[?]' or unresolved reference tokens found in entire document!")

print("\n" + "="*60)
print("CHECK 2: Front matter pagination & layout structure")
print("="*60)
for i in range(min(15, len(doc))):
    text = doc[i].get_text("text").strip()
    first_lines = [l.strip() for l in text.splitlines() if l.strip()][:3]
    # filter out non-ascii for display
    clean_lines = [' '.join(l.encode('ascii', 'replace').decode().split()) for l in first_lines]
    print(f"Page {i+1}: {' | '.join(clean_lines)}")

print("\n" + "="*60)
print("CHECK 3: Declaration page verification")
print("="*60)
decl_text = re.sub(r'\s+', ' ', re.sub(r'-\s+', '', doc[1].get_text("text")))
print("Declaration contains Kshitij Khede:", "Kshitij Khede" in decl_text or "KSHITIJ KHEDE" in decl_text)
print("Declaration contains 252IT010:", "252IT010" in decl_text)
print("Declaration contains Minor Project (IT897):", "Minor Project (IT897)" in decl_text)
print("Declaration contains Layout-Aware Hierarchical Knowledge Representation:", "Layout-Aware Hierarchical Knowledge Representation" in decl_text)
print("Declaration contains National Institute of Technology Karnataka:", "National Institute of Technology Karnataka" in decl_text)

print("\n" + "="*60)
print("CHECK 4: Certificate page verification")
print("="*60)
cert_text = re.sub(r'\s+', ' ', re.sub(r'-\s+', '', doc[2].get_text("text")))
print("Certificate contains JUNE 01, 2026 to AUGUST 15, 2026:", "JUNE 01, 2026 to AUGUST 15, 2026" in cert_text)
print("Certificate contains Minor Project (IT897):", "Minor Project (IT897)" in cert_text)
print("Certificate contains Layout-Aware Hierarchical Knowledge Representation:", "Layout-Aware Hierarchical Knowledge Representation" in cert_text)
print("Certificate contains Prof. ANANTHANARAYANA V. S.:", "ANANTHANARAYANA" in cert_text)
print("Certificate does NOT contain Chairman – DPGC:", "Chairman" not in cert_text and "DPGC" not in cert_text)

print("\n" + "="*60)
print("CHECK 5: Prohibited semester-phase terms check")
print("="*60)
prohibited = ["Phase 1", "Phase 2", "Phase I", "Phase II", "next semester", "future semester", "Phase-1", "Phase-2"]
found_prohibited = []
for page_num in range(len(doc)):
    text = doc[page_num].get_text("text")
    for term in prohibited:
        if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
            found_prohibited.append((page_num + 1, term))
if found_prohibited:
    print("Found prohibited terms:", found_prohibited)
else:
    print("SUCCESS: Zero prohibited project-division terms found!")

print("\n" + "="*60)
print("CHECK 6: Algorithm numbering check")
print("="*60)
algos_found = []
for page_num in range(len(doc)):
    text = doc[page_num].get_text("text")
    matches = re.findall(r'Algorithm \d+[:\s][^\n]+', text)
    if matches:
        algos_found.extend([(page_num + 1, m) for m in matches])
for p, a in algos_found:
    clean_a = a.encode('ascii', 'replace').decode()
    print(f"Page {p}: {clean_a}")

print("\n" + "="*60)
print("CHECK 7: Numerical results verification across chapters")
print("="*60)
full_text = ""
for page in doc:
    full_text += page.get_text("text") + "\n"

checks = [
    ("Naive inversion (21.8%)", "21.8%"),
    ("Column-aware inversion (0.5%)", "0.5%"),
    ("Naive monotonicity (78.2%)", "78.2%"),
    ("Column-aware monotonicity (99.5%)", "99.5%"),
    ("Numbered sections count (196)", "196"),
    ("Section numbering PAA (100.0)", "100.0"),
    ("PRE semantic PAA (51.5%)", "51.5%"),
    ("Lexical overlap PAA (34.7)", "34.7"),
    ("Sequential baseline PAA (17.9)", "17.9"),
    ("TreeRAG blocks extracted (262)", "262"),
    ("TreeRAG hierarchy edges (14)", "14"),
    ("TreeRAG runtime (1.8 s)", "1.8 s"),
    ("TreeRAG mean confidence (0.62)", "0.62"),
    ("Kahn damping factor (0.85)", "0.85"),
    ("Vertical line tolerance (20 pt)", "20 pt"),
]

all_passed = True
for label, val in checks:
    count = full_text.count(val)
    status = "PASS" if count > 0 else "FAIL"
    if count == 0:
        all_passed = False
    print(f"{label:35}: present {count:2d} times [{status}]")

print("\n" + "="*60)
if all_passed:
    print("ALL NUMERICAL VERIFICATION CHECKS PASSED!")
else:
    print("WARNING: Some numerical verification checks failed!")
print("="*60)

