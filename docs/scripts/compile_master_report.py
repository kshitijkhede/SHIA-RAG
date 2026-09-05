import os
import glob
from pathlib import Path

# Cross-platform: resolve paths relative to this script's location
_DOCS_DIR = Path(__file__).resolve().parent.parent
_ROOT_DIR = _DOCS_DIR.parent
chapters_dir = str(_DOCS_DIR / 'report_chapters')
output_file = str(_ROOT_DIR / 'SHIA_RAG_Complete_Report.md')

chapter_files = sorted(glob.glob(os.path.join(chapters_dir, '*.md')))
print(f"Found {len(chapter_files)} chapters to compile:")
for cf in chapter_files:
    print(" -", os.path.basename(cf))

compiled_content = []
for cf in chapter_files:
    with open(cf, 'r', encoding='utf-8') as f:
        compiled_content.append(f.read().strip())

full_text = '\n\n---\n\n'.join(compiled_content)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(full_text + '\n')

print(f"\nSuccessfully compiled {len(chapter_files)} chapters into {output_file}")
print(f"Total characters: {len(full_text)}")
print(f"Total lines: {len(full_text.splitlines())}")
