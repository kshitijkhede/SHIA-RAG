import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set high-DPI publication styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.0
plt.rcParams['axes.titlesize'] = 10.5
plt.rcParams['axes.labelsize'] = 9.5
plt.rcParams['figure.titlesize'] = 11.5
plt.rcParams['figure.dpi'] = 300

# Professional academic palette
C_PRIMARY = '#1f497d'      # Deep Navy
C_SECONDARY = '#2e75b6'    # Blue
C_ACCENT = '#008080'       # Teal
C_LIGHT = '#e8f1f5'        # Very soft blue
C_MUTED = '#595959'        # Neutral gray
C_ALERT = '#c00000'        # Crimson
C_SUCCESS = '#2e7d32'      # Forest Green
C_BOX_BG = '#f8fbfd'       # Off-white box fill
C_EDGE = '#2c3e50'         # Border line

# =============================================================================
# FIGURE 1: Overall System Architecture (Readable, Clean Boxes)
# =============================================================================
fig, ax = plt.subplots(figsize=(11.5, 4.5), dpi=300)
ax.axis('off')

stages = [
    ("Input PDF", "Multi-column digital\ntechnical doc"),
    ("Layout Ingestion", "PyMuPDF coordinate &\nbounding box parsing"),
    ("Reading Order", "2D Column-Aware\nGeometric Sorting"),
    ("Concept\nExtraction", "Heading Landmarks &\nEvidence Anchors"),
    ("Hierarchy\nInduction", "CPG++ Candidates &\nPRE Multi-Criteria Rank"),
    ("Semantic\nCross-Links", "Cosine Similarity &\nLateral Edge Discovery"),
    ("Validation & KCE", "Upward DFS Invariants &\nTopological Confidence"),
    ("Knowledge\nForest", "Rooted Tree Forest &\nSemantic Cross-Links")
]

box_w = 1.08
box_h = 0.85
spacing = 1.38

for i, (title, desc) in enumerate(stages):
    x = i * spacing + 0.1
    y = 0.75
    # Alternating subtle tint
    bg = '#dbe9f4' if i in [0, 7] else '#f0f4f8'
    edge = C_PRIMARY if i in [0, 7] else '#4a6572'
    lw = 2.0 if i in [0, 7] else 1.2
    
    rect = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.08,rounding_size=0.1",
                                  facecolor=bg, edgecolor=edge, linewidth=lw)
    ax.add_patch(rect)
    
    ax.text(x + box_w/2, y + box_h*0.64, title, ha='center', va='center',
            fontsize=9.0, fontweight='bold', color='#102a43')
    ax.text(x + box_w/2, y + box_h*0.25, desc, ha='center', va='center',
            fontsize=7.2, color='#334e68')
    
    if i < len(stages) - 1:
        ax.annotate('', xy=(x + box_w + 0.28, y + box_h/2), xytext=(x + box_w, y + box_h/2),
                    arrowprops=dict(arrowstyle="-|>", color=C_PRIMARY, lw=1.6, mutation_scale=12))

ax.set_xlim(-0.1, len(stages) * spacing)
ax.set_ylim(0.1, 2.0)
plt.title("End-to-End Functional Pipeline of SHIA-RAG Architecture", fontsize=11.5, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('images/fig1_overall_architecture.png', bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 2: 2D Geometric Reading Order Recovery Comparison
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

for ax in (ax1, ax2):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

# Left panel: Naive text extraction
ax1.set_title("Naive PDF Text Stream\n(Interleaved Column Scan: 28.6% Inversions)", fontsize=10, fontweight='bold', color=C_ALERT)
col1_y = [7.5, 5.5, 3.5, 1.5]
col2_y = [7.5, 5.5, 3.5, 1.5]

for idx, y in enumerate(col1_y):
    ax1.add_patch(patches.Rectangle((0.8, y), 3.8, 1.4, facecolor='#fdedec', edgecolor='#c0392b', lw=1.2))
    ax1.text(2.7, y + 0.7, f"Column 1 - Block {idx*2 + 1}", ha='center', va='center', fontsize=8, color='#900c3f')

for idx, y in enumerate(col2_y):
    ax1.add_patch(patches.Rectangle((5.4, y), 3.8, 1.4, facecolor='#fdedec', edgecolor='#c0392b', lw=1.2))
    ax1.text(7.3, y + 0.7, f"Column 2 - Block {idx*2 + 2}", ha='center', va='center', fontsize=8, color='#900c3f')

# Naive reading flow arrows (zigzagging across columns)
naive_pts = [(4.6, 8.2), (5.4, 8.2), (7.3, 7.5), (2.7, 6.9), (4.6, 6.2), (5.4, 6.2), (7.3, 5.5), (2.7, 4.9)]
for p in range(0, len(naive_pts)-1, 2):
    ax1.annotate('', xy=naive_pts[p+1], xytext=naive_pts[p],
                 arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.8, linestyle='--'))

# Right panel: Proposed 2D Geometric Sorting
ax2.set_title("SHIA-RAG Column-Aware 2D Geometric Recovery\n(Y-Tolerance Clustering: 100.0% Monotonic on Benchmark)", fontsize=10, fontweight='bold', color=C_SUCCESS)
for idx, y in enumerate(col1_y):
    ax2.add_patch(patches.Rectangle((0.8, y), 3.8, 1.4, facecolor='#e8f5e9', edgecolor='#27ae60', lw=1.2))
    ax2.text(2.7, y + 0.7, f"Block {idx + 1} (Col 1)", ha='center', va='center', fontsize=8, color='#1e824c')

for idx, y in enumerate(col2_y):
    ax2.add_patch(patches.Rectangle((5.4, y), 3.8, 1.4, facecolor='#e8f5e9', edgecolor='#27ae60', lw=1.2))
    ax2.text(7.3, y + 0.7, f"Block {idx + 5} (Col 2)", ha='center', va='center', fontsize=8, color='#1e824c')

# Ordered arrows
for i in range(len(col1_y)-1):
    ax2.annotate('', xy=(2.7, col1_y[i+1]+1.4), xytext=(2.7, col1_y[i]),
                 arrowprops=dict(arrowstyle="-|>", color=C_SUCCESS, lw=1.8))
# Column transition arrow
ax2.annotate('', xy=(7.3, 8.9), xytext=(2.7, 1.5),
             arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=1.8, linestyle=':'))
for i in range(len(col2_y)-1):
    ax2.annotate('', xy=(7.3, col2_y[i+1]+1.4), xytext=(7.3, col2_y[i]),
                 arrowprops=dict(arrowstyle="-|>", color=C_SUCCESS, lw=1.8))

plt.tight_layout()
plt.savefig('images/fig2_layout_reading_order.png', bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 3: Dual-Tier Data Representation Schema
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)

# Tier 1 Container
rect1 = patches.FancyBboxPatch((0.4, 0.4), 4.2, 5.2, boxstyle="round,pad=0.15",
                               facecolor='#f4f8fb', edgecolor=C_PRIMARY, lw=1.8)
ax.add_patch(rect1)
ax.text(2.5, 5.2, "Tier 1: Physical / Syntactic Model", ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=C_PRIMARY)

tier1_boxes = [
    ("StructuredDocument", "doc_id, total_pages, sha256_hash,\nordered_block_sequence"),
    ("TextBlock", "block_id, page_number, bbox (x0,y0,x1,y1),\nreading_order, font_size, font_weight"),
    ("BoundingBox", "x0, y0, x1, y1 (in points, 1/72 in)\nPhysical layout coordinate bounds")
]
for idx, (title, fields) in enumerate(tier1_boxes):
    y = 3.6 - idx * 1.5
    ax.add_patch(patches.Rectangle((0.7, y), 3.6, 1.15, facecolor='white', edgecolor='#7f8c8d', lw=1.0))
    ax.text(2.5, y + 0.82, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#2c3e50')
    ax.text(2.5, y + 0.38, fields, ha='center', va='center', fontsize=6.8, color='#595959')

# Tier 2 Container
rect2 = patches.FancyBboxPatch((5.4, 0.4), 4.2, 5.2, boxstyle="round,pad=0.15",
                               facecolor='#fbfcf7', edgecolor=C_SUCCESS, lw=1.8)
ax.add_patch(rect2)
ax.text(7.5, 5.2, "Tier 2: Semantic Knowledge Forest", ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=C_SUCCESS)

tier2_boxes = [
    ("ConceptNode", "canonical_name, definition, level (1..6),\nlocal_confidence, evidence_bbox_spans"),
    ("HierarchyEdge", "parent_id -> child_id, relation_type,\ntree_depth, acyclic_validation = True"),
    ("KnowledgeForest", "disjoint rooted trees G_hier = (V, E_hier)\n+ semantic cross-links E_cross")
]
for idx, (title, fields) in enumerate(tier2_boxes):
    y = 3.6 - idx * 1.5
    ax.add_patch(patches.Rectangle((5.7, y), 3.6, 1.15, facecolor='white', edgecolor='#7f8c8d', lw=1.0))
    ax.text(7.5, y + 0.82, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#2c3e50')
    ax.text(7.5, y + 0.38, fields, ha='center', va='center', fontsize=6.8, color='#595959')

# Grounding Anchor Arrow
ax.annotate('Grounding Evidence Span\n(Bounding Box Projection)',
            xy=(5.7, 4.0), xytext=(4.3, 4.0),
            ha='center', va='center', fontsize=7.5, fontweight='bold', color=C_ACCENT,
            arrowprops=dict(arrowstyle="<|-|>", color=C_ACCENT, lw=2.0, linestyle='--'))

plt.title("Two-Tier Knowledge Architecture: Syntactic Evidence Anchoring to Semantic Forest", fontsize=11, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('images/fig3_dual_tier_forest.png', bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 4: Hierarchy Induction Decision Pipeline (No hardcoded Figure number!)
# =============================================================================
fig, ax = plt.subplots(figsize=(10.5, 3.8), dpi=300)
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)

steps = [
    ("Child Concept\nc_i", "Extracted definition\nor section heading"),
    ("CPG++ Filter", "Order precedence\nlevel(u) < level(c_i)\nDoc isolation"),
    ("Candidate Set\nCPG(c_i)", "Set of candidate\nancestors {u_1, ...}"),
    ("PRE Scoring", "w_struct*S_struct +\nw_sem*S_sem +\nw_level*S_level"),
    ("Best Candidate\nu* = argmax", "Highest composite\nranking score"),
    ("Upward DFS\nValidation", "Check u* -> c_i\nPrevents cycles\nDepth <= 8"),
    ("Edge Insertion\nor Root", "Insert (u*, c_i)\nUpdate Tree Forest")
]

for i, (title, desc) in enumerate(steps):
    x = i * 1.65 + 0.3
    rect = patches.FancyBboxPatch((x, 1.0), 1.35, 2.0, boxstyle="round,pad=0.08",
                                  facecolor='#fcf3cf' if i == 3 else '#edf2f7',
                                  edgecolor='#d4ac0d' if i == 3 else '#4a6572',
                                  lw=1.5 if i in [3, 5] else 1.0)
    ax.add_patch(rect)
    ax.text(x + 0.67, 2.35, title, ha='center', va='center', fontsize=8, fontweight='bold', color='#1a202c')
    ax.text(x + 0.67, 1.5, desc, ha='center', va='center', fontsize=6.8, color='#4a5568')
    if i < len(steps) - 1:
        ax.annotate('', xy=(x + 1.65, 2.0), xytext=(x + 1.35, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=C_PRIMARY, lw=1.5))

plt.title("Hierarchical Parent Assignment Workflow via CPG++ and PRE Scoring", fontsize=10.5, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('images/fig4_hierarchy_induction.png', bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 5: Topological Confidence Propagation (KCE over E_hier)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

# Panel A: Topological Propagation Concept
ax1.axis('off')
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 5)
ax1.set_title("(a) Hierarchical Topological Traversal (over E_hier)", fontsize=10, fontweight='bold', color=C_PRIMARY)

# Root node
ax1.add_patch(patches.Circle((3.0, 4.2), 0.5, facecolor='#d4efdf', edgecolor='#27ae60', lw=1.5))
ax1.text(3.0, 4.2, "Root\nC=0.95", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#145a32')

# Level 1 nodes
ax1.add_patch(patches.Circle((1.6, 2.6), 0.45, facecolor='#e8f8f5', edgecolor='#16a085', lw=1.2))
ax1.text(1.6, 2.6, "Sec 1\nC=0.88", ha='center', va='center', fontsize=7, color='#0e6251')
ax1.add_patch(patches.Circle((4.4, 2.6), 0.45, facecolor='#e8f8f5', edgecolor='#16a085', lw=1.2))
ax1.text(4.4, 2.6, "Sec 2\nC=0.84", ha='center', va='center', fontsize=7, color='#0e6251')

# Edges to Level 1
ax1.annotate('', xy=(1.8, 3.0), xytext=(2.7, 3.8), arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=1.4))
ax1.annotate('', xy=(4.2, 3.0), xytext=(3.3, 3.8), arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=1.4))

# Level 2 nodes
ax1.add_patch(patches.Circle((0.9, 1.0), 0.4, facecolor='#fef9e7', edgecolor='#f39c12', lw=1.0))
ax1.text(0.9, 1.0, "Sub 1.1\nC=0.79", ha='center', va='center', fontsize=6.5, color='#7d6608')
ax1.add_patch(patches.Circle((2.3, 1.0), 0.4, facecolor='#fef9e7', edgecolor='#f39c12', lw=1.0))
ax1.text(2.3, 1.0, "Sub 1.2\nC=0.76", ha='center', va='center', fontsize=6.5, color='#7d6608')
ax1.add_patch(patches.Circle((4.4, 1.0), 0.4, facecolor='#fef9e7', edgecolor='#f39c12', lw=1.0))
ax1.text(4.4, 1.0, "Sub 2.1\nC=0.73", ha='center', va='center', fontsize=6.5, color='#7d6608')

# Edges to Level 2
ax1.annotate('', xy=(1.0, 1.4), xytext=(1.4, 2.2), arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=1.2))
ax1.annotate('', xy=(2.1, 1.4), xytext=(1.8, 2.2), arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=1.2))
ax1.annotate('', xy=(4.4, 1.4), xytext=(4.4, 2.1), arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=1.2))

# Horizontal cross-link (clearly labeled as excluded from KCE)
ax1.annotate('', xy=(3.9, 1.0), xytext=(2.8, 1.0),
             arrowprops=dict(arrowstyle="<->", color='#e74c3c', lw=1.4, linestyle='--'))
ax1.text(3.35, 1.25, "cross-link\n(excluded from KCE)", ha='center', va='bottom', fontsize=6.2, color='#c0392b')

# Panel B: Confidence Decay vs Depth under different Damping
depths = np.array([0, 1, 2, 3, 4, 5])
for delta in [0.70, 0.85, 0.95]:
    c_vals = [0.95]
    for d in depths[1:]:
        c_next = 0.88 * (delta + (1 - delta) * c_vals[-1])
        c_vals.append(c_next)
    ls = '-' if delta == 0.85 else ('--' if delta == 0.70 else ':')
    lw = 2.4 if delta == 0.85 else 1.5
    lbl = f"delta = {delta:.2f} (Selected)" if delta == 0.85 else f"delta = {delta:.2f}"
    ax2.plot(depths, c_vals, marker='o', label=lbl, linestyle=ls, linewidth=lw)

ax2.set_title("(b) Confidence Propagation vs Tree Depth", fontsize=10, fontweight='bold', color=C_PRIMARY)
ax2.set_xlabel("Hierarchy Depth Level (d)")
ax2.set_ylabel("Propagated Confidence C(v)")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()
ax2.set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig('images/fig5_kce_confidence_propagation.png', bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 6: Empirical Baseline Comparisons (Authoritative Table 4.3 match!)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

methods = ['Sequential\nNesting', 'Lexical\nPrefix', 'Syntactic\nHeading', 'SHIA-RAG\n(PRE)']
paa_scores = [44.2, 68.5, 77.4, 89.6]
nted_scores = [0.481, 0.672, 0.783, 0.884]

colors = ['#95a5a6', '#7f8c8d', '#2980b9', '#27ae60']

# Panel A: PAA
bars1 = ax1.bar(methods, paa_scores, color=colors, width=0.55, edgecolor='#2c3e50', lw=1.1)
ax1.set_title("(a) Parent Assignment Accuracy (PAA)", fontsize=10.5, fontweight='bold', color=C_PRIMARY)
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(0, 105)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1.8, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Panel B: NTED
bars2 = ax2.bar(methods, nted_scores, color=colors, width=0.55, edgecolor='#2c3e50', lw=1.1)
ax2.set_title("(b) Normalized Tree Edit Distance (NTED)", fontsize=10.5, fontweight='bold', color=C_PRIMARY)
ax2.set_ylabel("NTED Score")
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.suptitle("Hierarchy Induction Performance Across Baselines on CS-Textbook Benchmark", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('images/fig7_evaluation_metrics.png', bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 7: Reading-Order Recovery & Inversion Rate Comparison
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=300)

strategies = ['Naive Text\nStream', 'Top-Down\nCoordinate (y0)', 'SHIA-RAG\n2D Geometric']
rom_vals = [71.4, 83.2, 100.0]
inv_vals = [28.6, 16.8, 0.0]
strat_colors = ['#e74c3c', '#f39c12', '#27ae60']

bars_rom = ax1.bar(strategies, rom_vals, color=strat_colors, width=0.5, edgecolor='#2c3e50', lw=1.1)
ax1.set_title("(a) Reading-Order Monotonicity (ROM)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Monotonicity Rate (%)")
ax1.set_ylim(0, 115)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars_rom:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

bars_inv = ax2.bar(strategies, inv_vals, color=strat_colors, width=0.5, edgecolor='#2c3e50', lw=1.1)
ax2.set_title("(b) Reading-Order Inversion Error Rate", fontsize=10, fontweight='bold')
ax2.set_ylabel("Inversion Rate (%)")
ax2.set_ylim(0, 35)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars_inv:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.suptitle("Reading-Order Recovery Comparison Across Ingestion Strategies", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('images/fig6_treerag_case_study.png', bbox_inches='tight')
plt.close()

print("All publication-grade figures updated and re-generated successfully!")
