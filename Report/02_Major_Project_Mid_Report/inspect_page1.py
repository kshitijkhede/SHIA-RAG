import fitz

doc = fitz.open('report.pdf')
page = doc[0]
for b in page.get_text('dict')['blocks']:
    if 'lines' in b:
        for line in b['lines']:
            text = ' '.join(s['text'] for s in line['spans']).strip()
            if text:
                s0 = line['spans'][0]
                print(f"y={s0['bbox'][1]:.1f}, size={s0['size']:.1f}, font={s0['font']}, text={repr(text)}")
    elif b.get('type') == 1:
        print(f"IMAGE at bbox={b['bbox']}")
