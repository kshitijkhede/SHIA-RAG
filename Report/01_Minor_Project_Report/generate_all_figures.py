"""
SHIA-RAG Master Figure Generation Script
Generates publication-grade architectural diagrams (300 DPI)
perfectly synchronized with the report text, equations, and algorithms.

Figures generated:
1. images/fig1_overall_architecture.jpg & .png (Figure 3.1 in Report)
2. images/fig2_layout_reading_order.jpg & .png (Figure 3.2 in Report)
3. images/fig4_hierarchy_induction.jpg & .png (Figure 3.3 in Report)
4. images/fig5_kce_confidence_propagation.jpg & .png (Figure 3.4 in Report)
"""

import os
import sys
from pathlib import Path

# Auto-detect project virtualenv if matplotlib is not installed in current environment
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    for candidate in [
        Path(__file__).resolve().parents[2] / "shia-rag-minor" / ".venv" / "bin" / "python",
        Path(__file__).resolve().parents[2] / "shia-rag-core" / ".venv" / "bin" / "python",
    ]:
        if candidate.exists() and candidate != Path(sys.executable):
            import subprocess
            res = subprocess.run([str(candidate)] + sys.argv)
            sys.exit(res.returncode)
    raise


# Ensure target directory exists
os.makedirs('images', exist_ok=True)

# Publication styling defaults
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.5
plt.rcParams['figure.dpi'] = 300

# Refined academic color palette
C_NAVY = '#0f2942'
C_PRIMARY = '#1f497d'
C_BLUE = '#2e75b6'
C_LIGHT_BLUE = '#e8f0fe'
C_ACCENT = '#00796b'
C_TEAL_LIGHT = '#e0f2f1'
C_ALERT = '#c62828'
C_ALERT_LIGHT = '#ffebee'
C_SUCCESS = '#2e7d32'
C_SUCCESS_LIGHT = '#e8f5e9'
C_PURPLE = '#6a1b9a'
C_PURPLE_LIGHT = '#f3e5f5'
C_GOLD = '#b45309'
C_GOLD_LIGHT = '#fef3c7'
C_BG_CARD = '#f8fafc'
C_BORDER = '#94a3b8'
C_BORDER_DARK = '#334155'
C_TEXT_MAIN = '#0f172a'
C_TEXT_MUTED = '#475569'


# =============================================================================
# FIGURE 1: Overall System Architecture (Figure 3.1 in Report)
# Clean Serpentine 2-Row Flow:
# Row 1 (Left to Right): Input PDF -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4
#                                                                         |
# Row 2 (Right to Left): HKF Output <- Stage 8 <- Stage 7 <- Stage 6 <- Stage 5
# Zero crossing arrows, perfect logical ordering and visual clarity.
# =============================================================================
def generate_figure_1():
    print("Generating Figure 1: Overall Architecture (Serpentine Flow)...")
    fig, ax = plt.subplots(figsize=(19, 10.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10.5)

    # Main Canvas Header
    ax.text(9.5, 10.0, "SHIA-RAG: Layout-Aware Ingestion & Hierarchical Knowledge Forest Construction Pipeline",
            ha='center', va='center', fontsize=16, fontweight='bold', color=C_NAVY)
    ax.text(9.5, 9.65, "Eight Logical Processing Stages Flowing Sequentially from Multi-Column PDF Ingestion to Validated Knowledge Forest",
            ha='center', va='center', fontsize=11.5, color=C_TEXT_MUTED)

    # --- TOP ROW: Input + Stages 1 to 4 (Physical Extraction to Concept Extraction) ---
    # Coordinates for Row 1
    y_r1 = 5.7
    h_r1 = 3.3

    # Input Box
    ax.add_patch(patches.FancyBboxPatch((0.4, y_r1), 2.4, h_r1, boxstyle="round,pad=0.12",
                                        facecolor=C_LIGHT_BLUE, edgecolor=C_PRIMARY, lw=2.2))
    ax.add_patch(patches.FancyBboxPatch((0.4, y_r1 + h_r1 - 0.7), 2.4, 0.7, boxstyle="round,pad=0.06",
                                        facecolor=C_PRIMARY, edgecolor=C_PRIMARY, lw=1.0))
    ax.text(1.6, y_r1 + h_r1 - 0.35, "Input Document", ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(1.6, y_r1 + 1.4, "• Digital PDF\n• Multi-Column Layout\n• Embedded Fonts\n• Heading Hierarchy\n• Text & Figures",
            ha='center', va='center', fontsize=9, color=C_TEXT_MAIN, linespacing=1.35)

    # Arrow Input -> Stage 1
    ax.annotate('', xy=(3.3, y_r1 + h_r1/2), xytext=(2.8, y_r1 + h_r1/2),
                arrowprops=dict(arrowstyle="-|>", color=C_PRIMARY, lw=2.5, mutation_scale=16))

    row1_stages = [
        ("Stage 1: Ingestion", "PDFIngestionService",
         "• PyMuPDF page traversal\n• Extract text spans & bboxes\n• Font size & weight flags\n• Non-negative bounds check",
         C_PRIMARY, C_BG_CARD),
        ("Stage 2: Reading Order", "StructuralParser",
         "• 2D Gutter partitioning\n• Gutter split: x ≈ W / 2\n• Y-tolerance τ_y = 20 pt\n• Monotonic reading order",
         C_PRIMARY, C_BG_CARD),
        ("Stage 3: Classification", "StructuralParser",
         "• Body font estimation s_body\n• Numbered regex (^d+(.d+)*)\n• Typographic size ratios\n• Heading levels H1 to H6",
         C_PRIMARY, C_BG_CARD),
        ("Stage 4: Concept Extraction", "ConceptExtractor",
         "• Canonical concept labels\n• Copula definition regex\n• Physical bbox grounding\n• Evidence 7-tuple spans",
         C_PRIMARY, C_BG_CARD)
    ]

    w_box = 3.3
    x_r1_starts = [3.3, 7.2, 11.1, 15.0]

    for i, (title, module, desc, col_border, col_bg) in enumerate(row1_stages):
        x = x_r1_starts[i]
        rect = patches.FancyBboxPatch((x, y_r1), w_box, h_r1, boxstyle="round,pad=0.12",
                                      facecolor=col_bg, edgecolor=col_border, lw=2.0)
        ax.add_patch(rect)
        # Header strip
        ax.add_patch(patches.FancyBboxPatch((x, y_r1 + h_r1 - 0.7), w_box, 0.7, boxstyle="round,pad=0.06",
                                            facecolor=col_border, edgecolor=col_border, lw=1.0))
        ax.text(x + w_box/2, y_r1 + h_r1 - 0.25, title, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(x + w_box/2, y_r1 + h_r1 - 0.52, f"[{module}]", ha='center', va='center', fontsize=9, fontweight='bold', color='#bfdbfe')
        ax.text(x + 0.2, y_r1 + 1.25, desc, ha='left', va='center', fontsize=8.8, color=C_TEXT_MAIN, linespacing=1.35)

        # Arrow between stages in Row 1
        if i < len(row1_stages) - 1:
            ax.annotate('', xy=(x_r1_starts[i+1], y_r1 + h_r1/2), xytext=(x + w_box, y_r1 + h_r1/2),
                        arrowprops=dict(arrowstyle="-|>", color=col_border, lw=2.2, mutation_scale=16))

    # --- SERPENTINE CONNECTOR: Stage 4 (Row 1) Down to Stage 5 (Row 2) ---
    # From bottom of Stage 4 straight down to top of Stage 5!
    ax.annotate('', xy=(16.65, 4.3), xytext=(16.65, y_r1),
                arrowprops=dict(arrowstyle="-|>", color=C_GOLD, lw=3.0, mutation_scale=18))
    ax.text(16.75, 5.0, "Concepts V\n(Reading Order)", ha='left', va='center', fontsize=8.5, fontweight='bold', color=C_GOLD)

    # --- BOTTOM ROW: Stages 5 to 8 + Knowledge Forest Output (Right to Left) ---
    y_r2 = 0.8
    h_r2 = 3.5

    row2_stages = [
        ("Stage 5: Hierarchy Induction", "HierarchyBuilder",
         "• CPG++ candidate filtering\n• PRE 3-criteria scoring:\n  w_struct·S_struct + w_sem·S_sem\n  + w_level·S_level\n• Fallback root promotion",
         C_GOLD, '#fffbeb'),
        ("Stage 6: Cross-Linking", "HierarchyBuilder",
         "• Sentence-BERT bi-encoder\n• 384-dim normalized vectors\n• Cosine sim ≥ θ_cross = 0.75\n• Associative overlay E_cross",
         C_PURPLE, C_PURPLE_LIGHT),
        ("Stage 7: Graph Validation", "GraphValidator",
         "• Upward DFS cycle check\n• Single parent invariant\n• Depth bound D ≤ D_max = 8\n• Document isolation",
         C_SUCCESS, C_SUCCESS_LIGHT),
        ("Stage 8: Confidence (KCE)", "KCEScorer",
         "• Kahn's topological sort\n• Parent scored before child\n• Formula: C=C_loc·[δ+(1-δ)C_p]\n• Ancestral damping δ = 0.85",
         '#059669', '#ecfdf5')
    ]

    # Stage 5 is at the right (under Stage 4), Stage 6 to its left, Stage 7 to its left, Stage 8 to its left
    x_r2_starts = [15.0, 11.1, 7.2, 3.3]

    for i, (title, module, desc, col_border, col_bg) in enumerate(row2_stages):
        x = x_r2_starts[i]
        rect = patches.FancyBboxPatch((x, y_r2), w_box, h_r2, boxstyle="round,pad=0.12",
                                      facecolor=col_bg, edgecolor=col_border, lw=2.0)
        ax.add_patch(rect)
        # Header strip
        ax.add_patch(patches.FancyBboxPatch((x, y_r2 + h_r2 - 0.7), w_box, 0.7, boxstyle="round,pad=0.06",
                                            facecolor=col_border, edgecolor=col_border, lw=1.0))
        ax.text(x + w_box/2, y_r2 + h_r2 - 0.25, title, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(x + w_box/2, y_r2 + h_r2 - 0.52, f"[{module}]", ha='center', va='center', fontsize=9, fontweight='bold', color='#fef08a' if col_border == C_GOLD else '#ffffff')
        ax.text(x + 0.2, y_r2 + 1.35, desc, ha='left', va='center', fontsize=8.8, color=C_TEXT_MAIN, linespacing=1.35)

        # Arrow flowing left between stages in Row 2
        if i < len(row2_stages) - 1:
            ax.annotate('', xy=(x_r2_starts[i+1] + w_box, y_r2 + h_r2/2), xytext=(x, y_r2 + h_r2/2),
                        arrowprops=dict(arrowstyle="-|>", color=col_border, lw=2.2, mutation_scale=16))

    # --- FINAL OUTPUT: Hierarchical Knowledge Forest (Leftmost in Row 2) ---
    ax.add_patch(patches.FancyBboxPatch((0.4, y_r2), 2.4, h_r2, boxstyle="round,pad=0.12",
                                        facecolor='#dcfce7', edgecolor='#16a34a', lw=2.5))
    ax.add_patch(patches.FancyBboxPatch((0.4, y_r2 + h_r2 - 0.7), 2.4, 0.7, boxstyle="round,pad=0.06",
                                        facecolor='#16a34a', edgecolor='#16a34a', lw=1.0))
    ax.text(1.6, y_r2 + h_r2 - 0.35, "Output HKF", ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(1.6, y_r2 + 1.4, "• Strictly Acyclic Tree\n• G_hier = (V, E_hier)\n• Single Parent Invariant\n• Depth Bounded ≤ 8\n• Grounded BBox Spans\n• Confidence C ∈ [0, 1]",
            ha='center', va='center', fontsize=8.5, color=C_TEXT_MAIN, linespacing=1.3)

    # Arrow from Stage 8 (x=3.3) left to Output HKF (x=2.8)
    ax.annotate('', xy=(2.8, y_r2 + h_r2/2), xytext=(3.3, y_r2 + h_r2/2),
                arrowprops=dict(arrowstyle="-|>", color='#16a34a', lw=2.8, mutation_scale=18))

    plt.tight_layout()
    plt.savefig('images/fig1_overall_architecture.png', bbox_inches='tight')
    plt.savefig('images/fig1_overall_architecture.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
    plt.close()
    print("✓ Figure 1 saved successfully.")


# =============================================================================
# FIGURE 2: Reading-Order Recovery (Figure 3.2 in Report)
# Contrast between Naive 1D Stream & Column-Aware 2D Geometric Sorting
# Clean arrows with zero text collisions.
# =============================================================================
def generate_figure_2():
    print("Generating Figure 2: Reading Order Recovery...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.5), dpi=300)

    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

    # ---------------- LEFT PANEL: NAIVE STREAM INGESTION ----------------
    ax1.set_title("Naive 1D Text Stream Ingestion\n(Cross-Column Interleaving Failure)",
                  fontsize=13.5, fontweight='bold', color=C_ALERT, pad=12)

    # Page Container
    ax1.add_patch(patches.Rectangle((0.3, 0.4), 9.4, 8.8, facecolor='#fafafa', edgecolor='#b0bec5', lw=2.0))
    ax1.text(5.0, 9.0, "Two-Column Document Page (Physical Coordinate Space)", ha='center', va='center', fontsize=9.5, color=C_TEXT_MUTED)

    # Top Spanning Title Block
    ax1.add_patch(patches.Rectangle((1.0, 7.6), 8.0, 0.9, facecolor='#f1f5f9', edgecolor=C_BORDER, lw=1.5))
    ax1.text(5.0, 8.05, "Top Spanning Header / Title (Block 0)", ha='center', va='center',
             fontsize=10, fontweight='bold', color=C_TEXT_MAIN)

    left_y = [5.8, 4.0, 2.2]
    right_y = [5.8, 4.0, 2.2]

    # Draw boxes first
    for i, y in enumerate(left_y):
        ax1.add_patch(patches.Rectangle((1.0, y), 3.6, 1.3, facecolor=C_ALERT_LIGHT, edgecolor=C_ALERT, lw=1.6))
        ax1.text(2.8, y + 0.85, f"Left Column Block L{i+1}", ha='center', va='center', fontsize=10, fontweight='bold', color='#991b1b')
        ax1.text(2.8, y + 0.4, f"Physical BBox Coordinates\nStream Position {2*i + 1}", ha='center', va='center', fontsize=8, color='#7f1d1d')

    for i, y in enumerate(right_y):
        ax1.add_patch(patches.Rectangle((5.4, y), 3.6, 1.3, facecolor=C_ALERT_LIGHT, edgecolor=C_ALERT, lw=1.6))
        ax1.text(7.2, y + 0.85, f"Right Column Block R{i+1}", ha='center', va='center', fontsize=10, fontweight='bold', color='#991b1b')
        ax1.text(7.2, y + 0.4, f"Physical BBox Coordinates\nStream Position {2*i + 2}", ha='center', va='center', fontsize=8, color='#7f1d1d')

    # Zigzag arrows connecting between boxes (entering top / leaving bottom) without cutting text
    # Top -> L1
    ax1.annotate('', xy=(2.8, 7.1), xytext=(5.0, 7.6),
                 arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=2.0, linestyle='--'))
    # L1 -> R1 (horizontal between boxes)
    ax1.annotate('', xy=(5.4, 6.45), xytext=(4.6, 6.45),
                 arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=2.2, linestyle='--'))
    # R1 -> L2 (diagonal through gutter gap)
    ax1.annotate('', xy=(4.6, 4.65), xytext=(5.4, 5.8),
                 arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=2.2, linestyle='--',
                                 connectionstyle="arc3,rad=-0.15"))
    # L2 -> R2 (horizontal)
    ax1.annotate('', xy=(5.4, 4.65), xytext=(4.6, 4.65),
                 arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=2.2, linestyle='--'))
    # R2 -> L3 (diagonal through gutter gap)
    ax1.annotate('', xy=(4.6, 2.85), xytext=(5.4, 4.0),
                 arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=2.2, linestyle='--',
                                 connectionstyle="arc3,rad=-0.15"))
    # L3 -> R3 (horizontal)
    ax1.annotate('', xy=(5.4, 2.85), xytext=(4.6, 2.85),
                 arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=2.2, linestyle='--'))

    # Warning Badge
    ax1.text(5.0, 1.0, "Reading Order Inversion Rate: 28.6% (ROM = 71.4%)\nNaive stream interleaves left and right columns horizontally across gutter",
             ha='center', va='center', fontsize=9, fontweight='bold', color=C_ALERT,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#fff1f2', edgecolor=C_ALERT, lw=1.4))

    # ---------------- RIGHT PANEL: PROPOSED 2D GEOMETRIC SORTING ----------------
    ax2.set_title("Column-Aware 2D Geometric Sorting\n(Column Partitioning with Vertical Tolerance)",
                  fontsize=13.5, fontweight='bold', color=C_SUCCESS, pad=12)

    # Page Container
    ax2.add_patch(patches.Rectangle((0.3, 0.4), 9.4, 8.8, facecolor='#fafafa', edgecolor='#b0bec5', lw=2.0))
    ax2.text(5.0, 9.0, "Two-Column Document Page (Physical Coordinate Space)", ha='center', va='center', fontsize=9.5, color=C_TEXT_MUTED)

    # Gutter Guideline
    ax2.axvline(x=5.0, ymin=0.16, ymax=0.82, color='#94a3b8', linestyle=':', lw=1.8)
    ax2.text(5.0, 4.9, "Gutter Split\nx ≈ W / 2", ha='center', va='center', fontsize=7.8, color='#64748b',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='#ffffff', edgecolor='#cbd5e1', lw=1.0))

    # Top Spanning Title Block S_top
    ax2.add_patch(patches.Rectangle((1.0, 7.6), 8.0, 0.9, facecolor='#e0f2fe', edgecolor='#0284c7', lw=1.6))
    ax2.text(5.0, 8.05, "Top Spanning Header S_top (Rank 1 in Reading Order)\nwidth > 0.60·W, y0 < 0.30·H",
             ha='center', va='center', fontsize=9.2, fontweight='bold', color='#0369a1')

    # Arrow from S_top down to Left Column
    ax2.annotate('', xy=(2.8, 7.1), xytext=(2.8, 7.6),
                 arrowprops=dict(arrowstyle="-|>", color=C_SUCCESS, lw=2.2, mutation_scale=14))

    # Left Column Bucket C1
    for i, y in enumerate(left_y):
        ax2.add_patch(patches.Rectangle((1.0, y), 3.6, 1.3, facecolor=C_SUCCESS_LIGHT, edgecolor=C_SUCCESS, lw=1.6))
        ax2.text(2.8, y + 0.85, f"Block {i+2}: Left Column (C1)", ha='center', va='center', fontsize=10, fontweight='bold', color='#166534')
        ax2.text(2.8, y + 0.4, f"Y-tolerance clustering (|Δy| ≤ 20 pt)\nReading-Order Rank {i+2}", ha='center', va='center', fontsize=8, color='#14532d')
        if i < len(left_y) - 1:
            ax2.annotate('', xy=(2.8, left_y[i+1] + 1.3), xytext=(2.8, y),
                         arrowprops=dict(arrowstyle="-|>", color=C_SUCCESS, lw=2.0, mutation_scale=14))

    # Handover arrow: from bottom of Block 4, curving cleanly under the gutter and entering top of Block 5!
    # Does NOT cross through any box!
    ax2.annotate('', xy=(7.2, 7.1), xytext=(2.8, 2.2),
                 arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.6,
                                 connectionstyle="arc3,rad=-0.4", mutation_scale=16))
    ax2.text(5.0, 1.9, "Column Handover: C1 → C2", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_BLUE,
             bbox=dict(boxstyle="round,pad=0.2", facecolor='#ffffff', edgecolor=C_BLUE, lw=1.2))

    # Right Column Bucket C2
    for i, y in enumerate(right_y):
        ax2.add_patch(patches.Rectangle((5.4, y), 3.6, 1.3, facecolor=C_SUCCESS_LIGHT, edgecolor=C_SUCCESS, lw=1.6))
        ax2.text(7.2, y + 0.85, f"Block {i+5}: Right Column (C2)", ha='center', va='center', fontsize=10, fontweight='bold', color='#166534')
        ax2.text(7.2, y + 0.4, f"Sorted by ascending y0, then x0\nReading-Order Rank {i+5}", ha='center', va='center', fontsize=8, color='#14532d')
        if i < len(right_y) - 1:
            ax2.annotate('', xy=(7.2, right_y[i+1] + 1.3), xytext=(7.2, y),
                         arrowprops=dict(arrowstyle="-|>", color=C_SUCCESS, lw=2.0, mutation_scale=14))

    # Success Badge
    ax2.text(5.0, 1.0, "Reading-Order Monotonicity: 100.0% (Inversion Rate = 0.0%)\nStrict preservation of column reading flow: S_top ⊕ C1 ⊕ C2 ⊕ S_bot",
             ha='center', va='center', fontsize=9, fontweight='bold', color=C_SUCCESS,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#ecfdf5', edgecolor=C_SUCCESS, lw=1.4))

    plt.tight_layout()
    plt.savefig('images/fig2_layout_reading_order.png', bbox_inches='tight')
    plt.savefig('images/fig2_layout_reading_order.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
    plt.close()
    print("✓ Figure 2 saved successfully.")


# =============================================================================
# FIGURE 4: Hierarchy Induction Workflow (Figure 3.3 in Report)
# 6-Step Workflow with vertically balanced text and exact matching formulas
# =============================================================================
def generate_figure_4():
    print("Generating Figure 4: Hierarchy Induction Workflow...")
    fig, ax = plt.subplots(figsize=(18, 6.4), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6.4)

    ax.text(9.0, 6.0, "Hierarchical Parent Assignment Decision Pipeline (CPG++ Candidate Pruning & PRE Multi-Criteria)",
            ha='center', va='center', fontsize=14.5, fontweight='bold', color=C_NAVY)
    ax.text(9.0, 5.6, "Six-Stage Workflow Enforcing Document Isolation, Proximity-Semantic Ranking, and Acyclic Invariants",
            ha='center', va='center', fontsize=11, color=C_TEXT_MUTED)

    steps = [
        ("Step 1: Child Concept", "Incoming Concept Node c_i",
         "• Extracted in reading order\n• Attributes: name, level(c_i)\n• Reading rank pos(c_i)\n• 384d embedding vector e_ci\n• Linked evidence span bbox",
         C_PRIMARY, C_LIGHT_BLUE),
        ("Step 2: CPG++ Filter", "Candidate Generation",
         "• Strict Predecessor Set:\n  order(u) < order(c_i)\n  level(u) < level(c_i)\n  doc(u) == doc(c_i)\n• Fallback: all preceding u\n  if strict set is empty",
         C_BLUE, '#f0f9ff'),
        ("Step 3: PRE Scoring", "Multi-Criteria Evaluation",
         "• Score = w_struct·S_struct\n  + w_sem·S_sem + w_lvl·S_lvl\n• S_struct = 1/(1+ln(1+Δpos))\n• S_sem = max(0, cos(e_u, e_ci))\n• S_lvl = max(0, 1 - 0.2(Δl - 1))",
         C_GOLD, '#fffbeb'),
        ("Step 4: Optimal Parent", "Parent Selection",
         "• u* = argmax Score(u, c_i)\n• If candidate set empty:\n  Promote c_i to Root (depth 0)\n  (Guarantees 0% orphan rate)\n• If candidate exists: proceed",
         C_PURPLE, C_PURPLE_LIGHT),
        ("Step 5: Invariant Check", "Upward DFS Validation",
         "• Traversal: u* → parent(u*)...\n• Upward cycle detection\n• Verify depth(u*) + 1 ≤ D_max\n• Max depth D_max = 8 hops\n• Reject if cycle or depth limit",
         '#059669', '#ecfdf5'),
        ("Step 6: Tree Insertion", "Validated Edge (u*, c_i)",
         "• Insert edge into E_hier\n• Set depth(c_i) = depth(u*) + 1\n• Single parent invariant verified\n• Node attached to HKF forest\n• Ready for KCE propagation",
         C_SUCCESS, C_SUCCESS_LIGHT)
    ]

    box_w = 2.68
    box_h = 4.4
    x_starts = [0.4 + i * 2.92 for i in range(6)]
    y_pos = 0.6

    for i, (tag, title, desc, col_header, col_bg) in enumerate(steps):
        x = x_starts[i]
        rect = patches.FancyBboxPatch((x, y_pos), box_w, box_h, boxstyle="round,pad=0.1",
                                      facecolor=col_bg, edgecolor=col_header, lw=2.0)
        ax.add_patch(rect)
        # Header strip
        ax.add_patch(patches.FancyBboxPatch((x, y_pos + box_h - 0.8), box_w, 0.8, boxstyle="round,pad=0.06",
                                            facecolor=col_header, edgecolor=col_header, lw=1.0))
        ax.text(x + box_w/2, y_pos + box_h - 0.28, tag, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(x + box_w/2, y_pos + box_h - 0.58, title, ha='center', va='center', fontsize=8.8, fontweight='bold', color='#f8fafc')
        # Vertically centered description
        ax.text(x + 0.18, y_pos + 1.85, desc, ha='left', va='center', fontsize=8.6, color=C_TEXT_MAIN, linespacing=1.38)

        # Connector arrow
        if i < len(steps) - 1:
            ax.annotate('', xy=(x_starts[i+1], y_pos + box_h/2), xytext=(x + box_w, y_pos + box_h/2),
                        arrowprops=dict(arrowstyle="-|>", color=col_header, lw=2.4, mutation_scale=16))

    plt.tight_layout()
    plt.savefig('images/fig4_hierarchy_induction.png', bbox_inches='tight')
    plt.savefig('images/fig4_hierarchy_induction.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
    plt.close()
    print("✓ Figure 4 saved successfully.")


# =============================================================================
# FIGURE 5: Topological Confidence Propagation (Figure 3.4 in Report)
# Realistic Branching Tree Hierarchy + Matching Local Confidence Formula
# =============================================================================
def generate_figure_5():
    print("Generating Figure 5: KCE Confidence Propagation...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8.5), dpi=300)

    for ax in (ax1, ax2):
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    # ---------------- LEFT PANEL: FORMULATION & TOPOLOGICAL SORT ----------------
    ax1.set_title("Knowledge Confidence Estimation (KCE)\nvia Kahn-based Topological Propagation",
                  fontsize=13.5, fontweight='bold', color=C_PRIMARY, pad=12)

    # Topological Traversal Box
    ax1.add_patch(patches.FancyBboxPatch((0.4, 5.8), 9.2, 3.4, boxstyle="round,pad=0.12",
                                        facecolor=C_LIGHT_BLUE, edgecolor=C_PRIMARY, lw=2.0))
    ax1.text(5.0, 8.8, "Kahn's Topological Order Guarantee (Algorithm 3)",
             ha='center', va='center', fontsize=11, fontweight='bold', color=C_PRIMARY)
    kahn_desc = (
        "1. In-Degree Computation: in_degree(root) = 0; in_degree(child) = 1 in G_hier\n"
        "2. Queue Initialization: Q ← {v ∈ V | in_degree(v) = 0} (All root concepts)\n"
        "3. Topological Invariant: Dequeue parent u, compute C(u), propagate to children v,\n"
        "   decrement in_degree(v); when in_degree(v) = 0, push v to Q.\n"
        "4. Mathematical Guarantee: A parent node u is ALWAYS scored before its children."
    )
    ax1.text(0.7, 7.3, kahn_desc, ha='left', va='center', fontsize=8.8, color=C_TEXT_MAIN, linespacing=1.38)

    # Propagation Formula Box
    ax1.add_patch(patches.FancyBboxPatch((0.4, 1.0), 9.2, 4.4, boxstyle="round,pad=0.12",
                                        facecolor='#ecfdf5', edgecolor=C_SUCCESS, lw=2.2))
    ax1.text(5.0, 5.05, "KCE Local and Ancestral Confidence Formulas",
             ha='center', va='center', fontsize=11.5, fontweight='bold', color='#065f46')

    # Formula equations
    f_local = r"C_{local}(v) = \min(1.0, 0.50 + 0.15\cdot\mathbf{1}_{head} + 0.10\cdot\mathbf{1}_{bold} + 0.15\cdot\mathbf{1}_{num} + 0.10\cdot\mathbf{1}_{def})"
    f_prop = r"C(v) = C_{local}(v) \cdot [\delta + (1 - \delta) \cdot C(u)], \quad u = \text{Parent}(v)"

    ax1.text(5.0, 4.4, f"${f_local}$", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#047857')
    ax1.text(5.0, 3.75, f"${f_prop}$", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#15803d')

    f_desc = (
        "• Boundary Condition: Root nodes r have no parent → C(r) = C_local(r)\n"
        "• Damping Parameter: δ = 0.85 balances local evidence with ancestral reliability\n"
        "• Deterministic Heuristic: Confidence scores are bounded within [0, 1]\n"
        "• Single-Pass Linear Complexity: O(|V| + |E_hier|) via Kahn's queue traversal"
    )
    ax1.text(0.7, 2.3, f_desc, ha='left', va='center', fontsize=8.8, color=C_TEXT_MAIN, linespacing=1.4)

    # ---------------- RIGHT PANEL: BRANCHING TREE PROPAGATION ----------------
    ax2.set_title("Hierarchical Confidence Propagation on Document Subtree\n(Topological Parent-to-Child Downward Flow)",
                  fontsize=13.5, fontweight='bold', color=C_SUCCESS, pad=12)

    # Root Node: Level 0
    ax2.add_patch(patches.FancyBboxPatch((3.0, 8.0), 4.0, 1.3, boxstyle="round,pad=0.1",
                                        facecolor='#dcfce7', edgecolor=C_SUCCESS, lw=2.2))
    ax2.text(5.0, 8.85, "Root: 'TreeRAG Architecture'", ha='center', va='center', fontsize=10, fontweight='bold', color='#166534')
    ax2.text(5.0, 8.45, "Level 0 | in_degree = 0", ha='center', va='center', fontsize=8.5, color='#15803d')
    ax2.text(5.0, 8.15, "C(r) = C_local(r) = 0.950", ha='center', va='center', fontsize=9, fontweight='bold', color='#14532d')

    # Branch 1: Section 1 (Left Child of Root)
    ax2.add_patch(patches.FancyBboxPatch((0.5, 5.2), 4.0, 1.3, boxstyle="round,pad=0.1",
                                        facecolor='#ecfdf5', edgecolor='#059669', lw=1.8))
    ax2.text(2.5, 6.05, "1. 'Introduction'", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#065f46')
    ax2.text(2.5, 5.65, "Level 1 | C_local = 0.900", ha='center', va='center', fontsize=8.2, color='#047857')
    ax2.text(2.5, 5.35, "C = 0.90·[0.85 + 0.15(0.95)] = 0.893", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#064e3b')

    # Branch 2: Section 3 (Right Child of Root)
    ax2.add_patch(patches.FancyBboxPatch((5.5, 5.2), 4.0, 1.3, boxstyle="round,pad=0.1",
                                        facecolor='#ecfdf5', edgecolor='#059669', lw=1.8))
    ax2.text(7.5, 6.05, "3. 'System Architecture'", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#065f46')
    ax2.text(7.5, 5.65, "Level 1 | C_local = 0.920", ha='center', va='center', fontsize=8.2, color='#047857')
    ax2.text(7.5, 5.35, "C = 0.92·[0.85 + 0.15(0.95)] = 0.913", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#064e3b')

    # Connectors Root -> Sec 1 & Sec 3
    ax2.annotate('', xy=(2.5, 6.5), xytext=(4.0, 8.0),
                 arrowprops=dict(arrowstyle="-|>", color=C_SUCCESS, lw=2.2, mutation_scale=14))
    ax2.annotate('', xy=(7.5, 6.5), xytext=(6.0, 8.0),
                 arrowprops=dict(arrowstyle="-|>", color=C_SUCCESS, lw=2.2, mutation_scale=14))

    # Leaf under Section 1: 1.1 "Background"
    ax2.add_patch(patches.FancyBboxPatch((0.5, 2.5), 4.0, 1.3, boxstyle="round,pad=0.1",
                                        facecolor='#f0fdf4', edgecolor='#16a34a', lw=1.6))
    ax2.text(2.5, 3.35, "1.1 'Background & Motivation'", ha='center', va='center', fontsize=9, fontweight='bold', color='#15803d')
    ax2.text(2.5, 2.95, "Level 2 | C_local = 0.850", ha='center', va='center', fontsize=8.2, color='#166534')
    ax2.text(2.5, 2.65, "C = 0.85·[0.85 + 0.15(0.893)] = 0.836", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#14532d')

    ax2.annotate('', xy=(2.5, 3.8), xytext=(2.5, 5.2),
                 arrowprops=dict(arrowstyle="-|>", color='#16a34a', lw=2.0, mutation_scale=14))

    # Subsection under Section 3: 3.2 "Hierarchical Storage Tree"
    ax2.add_patch(patches.FancyBboxPatch((5.5, 2.5), 4.0, 1.3, boxstyle="round,pad=0.1",
                                        facecolor='#fefce8', edgecolor='#ca8a04', lw=1.8))
    ax2.text(7.5, 3.35, "3.2 'Hierarchical Storage Tree'", ha='center', va='center', fontsize=9, fontweight='bold', color='#854d0e')
    ax2.text(7.5, 2.95, "Level 2 | C_local = 0.900", ha='center', va='center', fontsize=8.2, color='#a16207')
    ax2.text(7.5, 2.65, "C = 0.90·[0.85 + 0.15(0.913)] = 0.888", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#713f12')

    ax2.annotate('', xy=(7.5, 3.8), xytext=(7.5, 5.2),
                 arrowprops=dict(arrowstyle="-|>", color='#ca8a04', lw=2.0, mutation_scale=14))

    # Leaf under 3.2: 3.2.1 "Tree Construction"
    ax2.add_patch(patches.FancyBboxPatch((5.5, 0.4), 4.0, 1.2, boxstyle="round,pad=0.1",
                                        facecolor='#fffbeb', edgecolor='#b45309', lw=1.6))
    ax2.text(7.5, 1.15, "3.2.1 'Tree Construction'", ha='center', va='center', fontsize=9, fontweight='bold', color='#92400e')
    ax2.text(7.5, 0.80, "Level 3 | C_local = 0.850", ha='center', va='center', fontsize=8.0, color='#9a3412')
    ax2.text(7.5, 0.55, "C = 0.85·[0.85 + 0.15(0.888)] = 0.836", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#7c2d12')

    ax2.annotate('', xy=(7.5, 1.6), xytext=(7.5, 2.5),
                 arrowprops=dict(arrowstyle="-|>", color='#b45309', lw=1.8, mutation_scale=14))

    plt.tight_layout()
    plt.savefig('images/fig5_kce_confidence_propagation.png', bbox_inches='tight')
    plt.savefig('images/fig5_kce_confidence_propagation.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
    plt.close()
    print("✓ Figure 5 saved successfully.")


if __name__ == '__main__':
    print("Generating all publication-grade figures for report...")
    generate_figure_1()
    generate_figure_2()
    generate_figure_4()
    generate_figure_5()
    print("\nFigures generated successfully in high-DPI (300 DPI) JPG & PNG formats!")
