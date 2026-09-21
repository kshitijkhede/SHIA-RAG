import fitz

doc = fitz.open('report.pdf')

for p in range(1, 6):
    print(f"==================== PAGE {p+1} ====================")
    page = doc[p]
    for b in page.get_text('dict')['blocks']:
        if 'lines' in b:
            for line in b['lines']:
                text = ' '.join(s['text'] for s in line['spans']).strip()
                if text:
                    s0 = line['spans'][0]
                    print(f"y={s0['bbox'][1]:.1f}, x={s0['bbox'][0]:.1f}, size={s0['size']:.1f}, font={s0['font']}, text={repr(text)}")
