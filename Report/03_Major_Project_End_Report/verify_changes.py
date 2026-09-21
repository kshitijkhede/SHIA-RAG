import fitz
doc = fitz.open('report_submission/main.pdf')

print('=== PAGE 2 (DECLARATION) ===')
p2 = doc[1]
for b in p2.get_text('dict')['blocks']:
    if 'lines' in b:
        for line in b['lines']:
            text = ' '.join(s['text'] for s in line['spans']).strip()
            if text and ('Khede' in text or 'Dept' in text or 'NITK' in text or 'Place' in text):
                print(f"  x={line['spans'][0]['bbox'][0]:.1f}, text={repr(text)}")

print('\n=== PAGE 3 (CERTIFICATE) ===')
p3 = doc[2]
for b in p3.get_text('dict')['blocks']:
    if 'lines' in b:
        for line in b['lines']:
            text = ' '.join(s['text'] for s in line['spans']).strip()
            if 'JUNE' in text or 'Duration' in text:
                print(f"  text={repr(text)}")

print('\n=== PAGE 11 (ABBREVIATIONS) ===')
p11 = doc[10]
print(f"  Drawings count on Page 11: {len(p11.get_drawings())}")
for b in p11.get_text('blocks')[:4]:
    print(f"    {repr(b[4][:50])}")
