import re
from pathlib import Path

# Cross-platform: resolve paths relative to root directory
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
md_file = str(_ROOT_DIR / 'SHIA_RAG_Complete_Report.md')
html_file = str(_ROOT_DIR / 'SHIA_RAG_Report.html')

with open(md_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Helper function to sanitize common inline math notation for clean WeasyPrint PDF output
def render_inline_math(match):
    m = match.group(1).strip()
    replacements = [
        (r'\\alpha', 'α'), (r'\\beta', 'β'), (r'\\gamma', 'γ'), (r'\\delta', 'δ'),
        (r'\\theta', 'θ'), (r'\\lambda', 'λ'), (r'\\mu', 'μ'), (r'\\pi', 'π'),
        (r'\\sigma', 'σ'), (r'\\tau', 'τ'), (r'\\phi', 'φ'), (r'\\omega', 'ω'),
        (r'\\mathcal\{V\}', '𝒱'), (r'\\mathcal\{E\}', 'ℰ'), (r'\\mathcal\{T\}', '𝒯'),
        (r'\\mathcal\{L\}', 'ℒ'), (r'\\mathcal\{C\}', '𝒞'), (r'\\mathcal\{H\}', 'ℋ'),
        (r'\\mathbf\{x\}', 'x'), (r'\\mathbf\{W\}', 'W'), (r'\\mathbf\{A\}', 'A'),
        (r'\\mathbf\{I\}', 'I'), (r'\\mathbf\{D\}', 'D'),
        (r'\\neq', '≠'), (r'\\ne', '≠'), (r'\\pm', '±'), (r'\\times', '×'),
        (r'\\leq', '≤'), (r'\\le', '≤'), (r'\\geq', '≥'), (r'\\ge', '≥'),
        (r'\\sum', '∑'), (r'\\prod', '∏'), (r'\\in', '∈'), (r'\\notin', '∉'),
        (r'\\forall', '∀'), (r'\\exists', '∃'), (r'\\cup', '∪'), (r'\\cap', '∩'),
        (r'\\subset', '⊂'), (r'\\subseteq', '⊆'), (r'\\to', '→'), (r'\\leftarrow', '←'),
        (r'\\implies', '⇒'), (r'\\land', '∧'), (r'\\lor', '∨'), (r'\\sim', '~'),
        (r'\\approx', '≈'), (r'\\cdot', '·'), (r'\\infty', '∞'), (r'\\sqrt', '√'),
        (r'\^2', '²'), (r'\^3', '³'),
        (r'\\text\{([^}]+)\}', r'\1'),
        (r'\\max', 'max'), (r'\\min', 'min'),
    ]
    for pattern, repl in replacements:
        m = re.sub(pattern, repl, m)
    m = re.sub(r'\\([a-zA-Z]+)', r'\1', m)
    return f'<span class="math-inline">{m}</span>'

# Convert $$ displayed math
def render_block_math(match):
    m = match.group(1).strip()
    replacements = [
        (r'\\alpha', 'α'), (r'\\beta', 'β'), (r'\\gamma', 'γ'), (r'\\delta', 'δ'),
        (r'\\theta', 'θ'), (r'\\lambda', 'λ'), (r'\\mu', 'μ'), (r'\\pi', 'π'),
        (r'\\sigma', 'σ'), (r'\\tau', 'τ'), (r'\\phi', 'φ'), (r'\\omega', 'ω'),
        (r'\\mathcal\{V\}', '𝒱'), (r'\\mathcal\{E\}', 'ℰ'), (r'\\mathcal\{T\}', '𝒯'),
        (r'\\mathcal\{L\}', 'ℒ'), (r'\\mathcal\{C\}', '𝒞'), (r'\\mathcal\{H\}', 'ℋ'),
        (r'\\mathbf\{x\}', 'x'), (r'\\mathbf\{W\}', 'W'), (r'\\mathbf\{A\}', 'A'),
        (r'\\mathbf\{I\}', 'I'), (r'\\mathbf\{D\}', 'D'),
        (r'\\neq', '≠'), (r'\\ne', '≠'), (r'\\pm', '±'), (r'\\times', '×'),
        (r'\\leq', '≤'), (r'\\le', '≤'), (r'\\geq', '≥'), (r'\\ge', '≥'),
        (r'\\sum', '∑'), (r'\\prod', '∏'), (r'\\in', '∈'), (r'\\notin', '∉'),
        (r'\\forall', '∀'), (r'\\exists', '∃'), (r'\\cup', '∪'), (r'\\cap', '∩'),
        (r'\\subset', '⊂'), (r'\\subseteq', '⊆'), (r'\\to', '→'), (r'\\leftarrow', '←'),
        (r'\\implies', '⇒'), (r'\\land', '∧'), (r'\\lor', '∨'), (r'\\sim', '~'),
        (r'\\approx', '≈'), (r'\\cdot', '·'), (r'\\infty', '∞'), (r'\\sqrt', '√'),
        (r'\^2', '²'), (r'\^3', '³'),
        (r'\\text\{([^}]+)\}', r'\1'),
        (r'\\max', 'max'), (r'\\min', 'min'),
    ]
    for pattern, repl in replacements:
        m = re.sub(pattern, repl, m)
    m = re.sub(r'\\([a-zA-Z]+)', r'\1', m)
    return f'<div class="math-block">{m}</div>'

# Apply math pre-rendering
text = re.sub(r'\$\$(.*?)\$\$', render_block_math, text, flags=re.DOTALL)
text = re.sub(r'(?<!\$)\$([^$\n]+)\$(?!\$)', render_inline_math, text)

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
  <title>SHIA-RAG: Complete Project Report & Master Specification</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #ffffff;
      --text: #1e293b;
      --text-muted: #64748b;
      --primary: #1e40af;
      --primary-dark: #1e3a8a;
      --border: #cbd5e1;
      --code-bg: #f1f5f9;
      --pre-bg: #0f172a;
      --callout-bg: #eff6ff;
      --callout-border: #2563eb;
    }}
    @page {{
      size: A4;
      margin: 18mm 15mm 18mm 15mm;
      @top-left {{
        content: "SHIA-RAG: Semantic Hierarchy Induction Architecture for RAG";
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #94a3b8;
      }}
      @top-right {{
        content: "NITK Surathkal — M.Tech Report";
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #94a3b8;
      }}
      @bottom-right {{
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #64748b;
      }}
    }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.6;
      color: var(--text);
      background: var(--bg);
      max-width: 960px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, h3, h4, h5, h6 {{
      color: #0f172a;
      font-weight: 700;
      line-height: 1.25;
    }}
    h1 {{
      font-size: 2.1rem;
      border-bottom: 2px solid var(--primary);
      padding-bottom: 10px;
      margin-top: 18px;
      letter-spacing: -0.02em;
    }}
    h2 {{
      font-size: 1.5rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 6px;
      margin-top: 36px;
      letter-spacing: -0.01em;
    }}
    h3 {{
      font-size: 1.18rem;
      margin-top: 24px;
    }}
    table {{
      width: 100%;
      max-width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 0.82rem;
      table-layout: auto;
      word-wrap: break-word;
    }}
    th, td {{
      padding: 6px 8px;
      border: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
      word-break: break-word;
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
      font-size: 0.82em;
      background: var(--code-bg);
      border: 1px solid #cbd5e1;
      padding: 2px 4px;
      border-radius: 4px;
      color: #0f172a;
      word-break: break-word;
    }}
    pre {{
      background: var(--pre-bg);
      color: #f8fafc;
      padding: 14px 16px;
      border-radius: 6px;
      overflow-x: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.80rem;
      line-height: 1.45;
    }}
    pre code {{
      background: transparent;
      border: none;
      color: inherit;
      padding: 0;
    }}
    blockquote {{
      margin: 18px 0;
      padding: 12px 16px;
      background: var(--callout-bg);
      border-left: 4px solid var(--callout-border);
      border-radius: 0 4px 4px 0;
      color: #1e3a8a;
      font-size: 0.92rem;
    }}
    blockquote p {{
      margin: 0;
    }}
    hr {{
      border: none;
      border-top: 1px solid var(--border);
      margin: 32px 0;
    }}
    .math-inline {{
      font-family: 'Inter', serif;
      font-style: italic;
      color: #1e293b;
      background: #f8fafc;
      padding: 1px 4px;
      border-radius: 3px;
    }}
    .math-block {{
      display: block;
      text-align: center;
      margin: 14px 0;
      padding: 10px 14px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      font-family: 'Inter', serif;
      font-style: italic;
      color: #0f172a;
      font-size: 1.02rem;
    }}
    .mermaid {{
      margin: 20px 0;
      text-align: center;
      background: #ffffff;
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    @media print {{
      body {{
        max-width: 100%;
        margin: 0;
        padding: 0;
        font-size: 9.5pt;
        line-height: 1.5;
      }}
      h1 {{
        font-size: 1.7rem;
        page-break-before: always;
        page-break-after: avoid;
      }}
      h1:first-of-type {{
        page-break-before: avoid;
      }}
      h2 {{
        font-size: 1.3rem;
        page-break-after: avoid;
      }}
      h3 {{
        font-size: 1.1rem;
        page-break-after: avoid;
      }}
      table {{
        font-size: 7.5pt;
        page-break-inside: auto;
      }}
      tr {{
        page-break-inside: avoid;
        page-break-after: auto;
      }}
      th, td {{
        padding: 4px 6px;
      }}
      pre {{
        font-size: 7.2pt;
        padding: 8px 10px;
        white-space: pre-wrap;
        word-break: break-all;
      }}
      blockquote {{
        page-break-inside: avoid;
        font-size: 8.5pt;
        padding: 8px 12px;
      }}
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
