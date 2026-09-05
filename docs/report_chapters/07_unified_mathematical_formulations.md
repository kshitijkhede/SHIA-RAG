## 7. Unified Mathematical Formulations

### 7.1 Global Hierarchy Energy Minimization Function

The structural optimality of the entire Knowledge Forest $\mathcal{F} = (\mathcal{V}, \mathcal{E}_{\text{hier}})$ is governed by the global energy objective $J(\mathcal{F})$:

$$J(\mathcal{F}) = \sum_{(P, N) \in \mathcal{E}_{\text{hier}}} E(P, N) + \mu \cdot R(\mathcal{F})$$

Where the pairwise edge energy $E(P, N)$ is defined as:

$$E(P, N) = \lambda_1 (1 - \cos(E_P, E_N)) + \lambda_2 \cdot \frac{\text{depth}(N)}{D_{\max}} + \lambda_3 (1 - \text{Conf}(P, N)) + \lambda_4 \cdot \mathbb{1}[\text{Type}(P) \not\succ \text{Type}(N)]$$

And the global tree regularization term $R(\mathcal{F})$ penalizes depth variance across leaf nodes:

$$R(\mathcal{F}) = \frac{1}{|\text{Leaves}(\mathcal{F})|} \sum_{l \in \text{Leaves}} (\text{depth}(l) - \bar{d})^2$$

**Canonical Parameter Defaults:**  
$\lambda_1 = 0.35$ (Semantic affinity), $\lambda_2 = 0.15$ (Depth regularization), $\lambda_3 = 0.25$ (Confidence preservation), $\lambda_4 = 0.25$ (Ontological validity), $\mu = 0.10$, $D_{\max} = 8$.

---

### 7.2 PRE Unified Parent Ranking Score

When evaluating candidate parent $P$ for new concept $N$, PRE maximizes the unified fitness score:

$$\text{Score}(P, N) = w_1 \cdot \cos(E_P, E_N) + w_2 \cdot H(P, N) + w_3 \cdot \frac{1}{\text{depth}(P) + 1} + w_4 \cdot S_{\text{evi}}(P, N) + w_5 \cdot \text{Conf}(P)$$

Where:
* $\cos'(E_P, E_N) = \frac{\cos(E_P, E_N) + 1}{2} \in [0, 1]$: Cosine similarity of dense embeddings, rescaled from $[-1, 1]$ to $[0, 1]$ to ensure non-negative contribution.
* $H(P, N) \in \{0, 1\}$: Hypernym indicator ($1$ if Hearst patterns or LLM verify that $N$ is-a $P$).
* $\frac{1}{\text{depth}(P) + 1}$: Inherent bias favoring shallower, broader conceptual parents.
* $S_{\text{evi}}(P, N) \in [0, 1]$: Empirical co-occurrence frequency of $N$ within sections governed by $P$.
* $\text{Conf}(P) \in [0, 1]$: Pre-existing confidence score of parent node $P$.
* **Reconciled Weight Vector:** $w_1 = 0.30, w_2 = 0.25, w_3 = 0.15, w_4 = 0.20, w_5 = 0.10$ (satisfying $\sum_{i=1}^5 w_i = 1.0$).

---

### 7.3 Formal DAG Precedence-Constrained Knapsack Formulation

Given retrieved subgraph $\mathcal{G}' = (\mathcal{V}', \mathcal{E}'_{\text{hier}})$:

$$\max_{\mathbf{x}} \sum_{i \in \mathcal{V}'} \left( \text{Rel}(v_i, q) \cdot \text{Conf}(v_i) \right) x_i$$
$$\text{subject to} \quad \sum_{i \in \mathcal{V}'} \text{Tokens}(v_i) \cdot x_i \le B$$
$$x_i \le x_p \quad \forall p \in \text{Parents}(i) \quad \text{in } \mathcal{E}'_{\text{hier}}$$
$$x_i \in \{0, 1\} \quad \forall i \in \mathcal{V}'$$

---

### 7.4 Thompson Sampling Dynamics & Graph Laplacian Diffusion

1. **Posterior Sampling:**
   $$\tilde{w}_e \sim \text{Beta}(\alpha_e, \beta_e), \qquad \mathbb{E}[\tilde{w}_e] = \frac{\alpha_e}{\alpha_e + \beta_e}$$
2. **Bayesian Posterior Update:**
   $$\alpha_e \leftarrow \alpha_e + R, \qquad \beta_e \leftarrow \beta_e + (1 - R) \quad \forall e \in \text{TraversalPath}$$
3. **Graph Laplacian Regularization:**
   Let $\mathbf{A}$ be the adjacency matrix of expected edge weights $A_{uv} = \frac{\alpha_{uv}}{\alpha_{uv} + \beta_{uv}}$, and $\mathbf{D}$ the degree matrix $D_{uu} = \sum_v A_{uv}$. The symmetric normalized Laplacian is:
   $$\mathcal{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$$
   The smooth diffused weight matrix $\mathbf{W}^{(t+1)}$ is computed as:
   $$\mathbf{W}^{(t+1)} = (1 - \gamma) \mathbf{A} + \gamma (\mathbf{I} - \mathcal{L}_{\text{sym}}) \mathbf{A}$$
   Where $\gamma = 0.05$ prevents over-smoothing while diffusing empirical path rewards to adjacent conceptual neighbors.

---

### 7.5 Query Complexity Scoring & Depth Allocation

The query complexity score $\Psi(q)$ determines the allocated depth $\tau$:

$$\Psi(q) = \omega_e \cdot \min(1.0, \frac{|E_q|}{3}) + \omega_c \cdot \mathbb{I}[q \text{ contains comparative lexemes}] + \omega_m \cdot \min(1.0, \frac{\text{EstHops}(q)}{3})$$

Where $\omega_e = 0.35, \omega_c = 0.35, \omega_m = 0.30$. The assigned traversal depth is:
$$\tau = \begin{cases} 
0 & \text{if } |E_q| = 0 \quad (\text{Mode 4: Parametric / No Retrieval}) \\
1 & \text{if } \Psi(q) < 0.30 \quad (\text{Mode 1: Thematic / Broad Overview}) \\
1 & \text{if } 0.30 \le \Psi(q) < 0.55 \land |E_q| = 1 \quad (\text{Mode 2: Local Factual Needle}) \\
3 & \text{if } 0.55 \le \Psi(q) < 0.85 \quad (\text{Mode 3: Multi-Hop Comparative}) \\
4 & \text{if } \Psi(q) \ge 0.85 \quad (\text{Mode 3: Deep Chain of Reasoning})
\end{cases}$$
