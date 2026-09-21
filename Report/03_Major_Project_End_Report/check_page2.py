import fitz
doc = fitz.open('report.pdf')
page = doc[1]
for b in page.get_text('dict')['blocks']:
    if 'lines' in b:
        for line in b['lines']:
            text = ' '.join(s['text'] for s in line['spans']).strip()
            if text:
                print(f"x={line['spans'][0]['bbox'][0]:.1f}, y={line['spans'][0]['bbox'][1]:.1f}, text={repr(text)}")
