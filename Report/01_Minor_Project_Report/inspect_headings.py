import fitz
from pathlib import Path

pdf_path = "main.pdf" if Path("main.pdf").exists() else "report.pdf"
doc = fitz.open(pdf_path)

with open('headings_report.txt', 'w', encoding='utf-8') as out:
    for p in range(len(doc)):
        page = doc[p]
        blocks = page.get_text('dict')['blocks']
        has_heading = False
        for b in blocks:
            if 'lines' in b:
                for line in b['lines']:
                    for span in line['spans']:
                        text = span['text'].strip()
                        if not text:
                            continue
                        if span['size'] > 12.5 or 'CHAPTER' in text:
                            if not has_heading:
                                out.write(f"\n=== Page {p+1} (type={page.rect}) ===\n")
                                has_heading = True
                            out.write(f"  size={span['size']:.1f}, font={span['font']}, y={span['bbox'][1]:.1f}, text={repr(text)}\n")

print("Done writing headings_report.txt")
