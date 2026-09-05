import re
from pathlib import Path

# Cross-platform: resolve paths relative to root directory
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
md_file = str(_ROOT_DIR / 'SHIA_RAG_Complete_Report.md')
html_file = str(_ROOT_DIR / 'SHIA_RAG_Report.html')

with open(md_file, 'r', encoding='utf-8') as f:
    text = f.read()

try:
    import markdown
    html_body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc', 'sane_lists'])
except ImportError:
    import markdown2
    html_body = markdown2.markdown(text, extras=['tables', 'fenced-code-blocks', 'toc'])

# Wrap mermaid code blocks for mermaid.js
html_body = re.sub(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>',
    r'<div class="mermaid">\1</div>',
    html_body,
    flags=re.DOTALL
)

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SHIA-RAG 2.0: Master Specification & Complete Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
  </script>
  <style>
    :root {{
      --bg: #ffffff;
      --text: #1e293b;
      --text-muted: #64748b;
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --border: #e2e8f0;
      --code-bg: #f8fafc;
      --pre-bg: #0f172a;
      --callout-bg: #eff6ff;
      --callout-border: #3b82f6;
    }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.7;
      color: var(--text);
      background: var(--bg);
      max-width: 1020px;
      margin: 0 auto;
      padding: 40px 24px;
    }}
    h1, h2, h3, h4, h5, h6 {{
      color: #0f172a;
      font-weight: 700;
      line-height: 1.3;
    }}
    h1 {{
      font-size: 2.3rem;
      border-bottom: 2px solid var(--primary);
      padding-bottom: 12px;
      margin-top: 20px;
      letter-spacing: -0.02em;
    }}
    h2 {{
      font-size: 1.65rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      margin-top: 48px;
      letter-spacing: -0.01em;
    }}
    h3 {{
      font-size: 1.25rem;
      margin-top: 32px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 24px 0;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 10px 14px;
      border: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f1f5f9;
      font-weight: 600;
      color: #0f172a;
    }}
    tr:nth-child(even) {{
      background: #f8fafc;
    }}
    code {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.88em;
      background: var(--code-bg);
      border: 1px solid #cbd5e1;
      padding: 2px 6px;
      border-radius: 4px;
      color: #0f172a;
    }}
    pre {{
      background: var(--pre-bg);
      color: #f8fafc;
      padding: 18px 20px;
      border-radius: 8px;
      overflow-x: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.88rem;
      line-height: 1.5;
    }}
    pre code {{
      background: transparent;
      border: none;
      color: inherit;
      padding: 0;
    }}
    blockquote {{
      margin: 24px 0;
      padding: 16px 20px;
      background: var(--callout-bg);
      border-left: 4px solid var(--callout-border);
      border-radius: 0 6px 6px 0;
      color: #1e3a8a;
    }}
    blockquote p {{
      margin: 0;
    }}
    hr {{
      border: none;
      border-top: 1px solid var(--border);
      margin: 40px 0;
    }}
    .mermaid {{
      margin: 28px 0;
      text-align: center;
      background: #ffffff;
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    @media print {{
      body {{ max-width: 100%; padding: 10px; font-size: 11pt; }}
      h1, h2 {{ page-break-after: avoid; }}
      table, pre, blockquote {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_template)

print(f"Successfully rendered HTML report to {html_file}")
print(f"File size: {len(html_template)} bytes")
