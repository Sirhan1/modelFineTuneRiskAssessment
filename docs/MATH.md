# Math and Logic Specification

This document makes the implemented math and decision logic explicit for
`alignment-risk`.

Primary references:

- **[AIC-2026]** *The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety*  
  PDF: `https://arxiv.org/pdf/2602.15799`
- **[ALIGNGUARD-2025]** *AlignGuard-LoRA: Alignment-Preserving Fine-Tuning via Fisher-Guided Decomposition and Riemannian-Geodesic Collision Regularization*  
  PDF: `https://arxiv.org/pdf/2508.02079`

For module-to-paper section mapping, see `docs/SOURCES.md`.

## 1. Notation

- Model parameters (selected trainable subset): $\theta \in \mathbb{R}^n$.
- Per-sample safety loss: $\ell_s(\theta; x_i)$.
- Fine-tuning loss: $\ell_f(\theta; b)$.
- Per-sample safety gradient: $g_i = \nabla_\theta \ell_s(\theta; x_i)$.
- Gradient matrix: $G \in \mathbb{R}^{m \times n}$, row $i$ is $g_i^\top$.
- Empirical Fisher approximation: $F \approx \frac{1}{m}G^\top G$.
- Fisher subspace basis: $B \in \mathbb{R}^{n \times d}$ (columns orthonormal).
- Top Fisher eigenvalues: $\lambda_1,\dots,\lambda_d \ge 0$.
- Initial update estimate from one fine-tune batch: $\Delta\theta_0$.
- Number of batches used in curvature averaging: $N_b$.
- Learning rate: $\eta$.
- Time/step variable in forecast: $t$ (derived from step index and `step_size`).

## 2. Parameter Selection Logic

The implementation first selects trainable parameters, optionally by explicit
allowlist:

- `full` mode: use all selected trainable parameters.
- `lora` mode:
  - if `fisher.parameter_names` is provided, that allowlist is used directly in
    the caller-provided order;
  - explicit allowlists are validated strictly: unknown, non-trainable, or
    duplicate names raise;
  - otherwise names are filtered by markers (`lora_`, `lora_A`, `lora_B`);
  - if no LoRA match and `require_lora_match=True`, raise.
  - if no LoRA match and `require_lora_match=False`, return a skipped report.

## 3. Vectorization Utilities

- Parameters are flattened by concatenation in a fixed order.
- If a gradient/tensor is `None`, it is replaced with zeros of matching shape.
- Length mismatch between tensor list and reference parameter list raises.
- Batch helpers recursively:
  - move tensors to a target device;
  - infer batch size from first non-scalar tensor encountered;
  - preserve scalar metadata tensors (do not slice them).

## 4. Fisher Estimation

### 4.1 Per-sample gradient collection

Given `max_samples = m`:

- **loop backend**:
  - iterates over each sample in each batch;
  - computes $\nabla_\theta \ell_s$ via standard backward/autograd.
- **vmap backend**:
  - builds functional model call and vectorized gradient function;
  - computes per-sample gradients in chunks (`vmap_chunk_size`).
- **auto backend**:
  - on first batch, compares loop vs vmap probe gradients (`auto_probe_*`);
  - if close, uses vmap; otherwise warns and uses loop fallback.

All collected gradients are promoted to float32 before CPU-side spectral ops
when needed (fp16/bf16 safety).

### 4.2 Fisher diagonal and module scores

With gradient matrix $G$:

$$
\operatorname{diag}(F)_j = \frac{1}{m}\sum_{i=1}^{m} G_{ij}^2
$$

Module score for module $M$:

$$
s_M =
\frac{1}{|J_M|}\sum_{j \in J_M}\operatorname{diag}(F)_j
$$

where $J_M$ are flattened parameter indices for that module prefix, and
$s_M$ corresponds to `module_score(M)` in code.

Top sensitive weights are the top-`k` entries of $\operatorname{diag}(F)$.

### 4.3 Subspace extraction

Config option `subspace_method`:

1. **`svd`**
   - $G = U\Sigma V^\top$
   - eigenvalues of $\frac{1}{m}G^\top G$: $\lambda_i = \sigma_i^2/m$
   - basis columns: first $d$ columns of $V$.

2. **`randomized_svd`**
   - draw Gaussian sketch $\Omega \in \mathbb{R}^{n \times k}$;
   - $Y = G\Omega$, optional power iterations $Y \leftarrow G(G^\top Y)$;
   - QR: $Y = QR$;
   - reduced matrix: $\widetilde{G} = Q^\top G$;
   - SVD of $\widetilde{G}$ gives approximate right singular vectors/eigenpairs.

3. **`diag_topk`**
   - choose top-$d$ diagonal indices from $\operatorname{diag}(F)$;
   - basis is canonical one-hot columns at those indices.

### 4.4 Rank selection by explained variance

Let candidate eigenvalues be $\lambda_1,\dots,\lambda_r$ after method-specific
preselection.

- If `target_explained_variance is None`: keep `max_rank`.
- If target $\le 0$: keep 1.
- If target $\ge 1$: keep `max_rank`.
- If clipped total energy $\sum_{i=1}^{r}\max(\lambda_i,0)\le 0$: keep `max_rank`.
- Else choose smallest $d$ such that:

$$
\frac{\sum_{i=1}^{d}\max(\lambda_i,0)}
{\sum_{i=1}^{r}\max(\lambda_i,0)}
\ge \tau
$$

where $\tau$ is `target_explained_variance`.

## 5. Initial Overlap Risk

Given update vector $\Delta\theta_0$ and subspace basis $B$:

$$
P_M(\Delta\theta_0)=BB^\top \Delta\theta_0
$$

$$
r_{\mathrm{proj}}=
\frac{\|P_M(\Delta\theta_0)\|_2}{\|\Delta\theta_0\|_2+\varepsilon}
$$

with $\varepsilon=10^{-12}$ for numerical safety and clamped to $[0,1]$.

`trigger_curvature_check` is true when:

$$
r_{\mathrm{proj}} \le \tau_{\mathrm{orth}}
$$

where $\tau_{\mathrm{orth}}$ corresponds to `orthogonality_threshold`.

## 6. Curvature Coupling

Sample-weighted mean fine-tuning loss over up to `max_batches`:

$$
L_f(\theta)=
\frac{\sum_{b=1}^{N_b} n_b\,\bar{\ell}_f(\theta; b)}
{\sum_{b=1}^{N_b} n_b}
$$

where $\bar{\ell}_f(\theta; b)$ is the mean loss of batch $b$ and $n_b$ is that
batch's sample count.

Then:

$$
g=\nabla_\theta L_f(\theta)
$$

Directional Hessian-vector product is computed as:

$$
a = H g
$$

via autograd on $\langle g, \operatorname{stopgrad}(g)\rangle$.

Fisher-weighted overlap magnitudes:

$$
\hat{\epsilon}=\sqrt{\sum_{i=1}^{d}\lambda_i\langle b_i,g\rangle^2}
$$

$$
\hat{\gamma}=\sqrt{\sum_{i=1}^{d}\lambda_i\langle b_i,a\rangle^2}
$$

Also returned: $\|a\|_2$ and $\|P_M(a)\|_2$.

## 7. Forecast Model

Pipeline scales local terms to step-scale terms:

$$
\epsilon = \eta \hat{\epsilon},\qquad \gamma = \eta^2 \hat{\gamma}
$$

Stable curvature floor used as effective $\lambda_{\min}$:

$$
\lambda_{\min}=\operatorname{quantile}_{0.1}
(\max(\lambda_i,0))
$$

Forecast equations (`times = steps * step_size`):

$$
\text{drift}(t)=\max\left(\frac{\gamma}{2}t^2-\epsilon t,\;0\right)
$$

$$
\widehat{\Delta u}(t)=\frac{1}{2}\lambda_{\min}\,\text{drift}(t)^2
$$

$$
\text{quartic}(t)=\frac{\lambda_{\min}\gamma^2}{8}t^4
$$

Collapse step is the first index where:

$$
\widehat{\Delta u}(t)\ge L_{\mathrm{collapse}}
$$

where $L_{\mathrm{collapse}}$ corresponds to `collapse_loss_threshold`.

## 8. Adaptive Curvature Refinement Logic

If enabled and `adaptive_curvature_max_batches > curvature.max_batches`:

- If no collapse yet, refine when terminal estimated loss exceeds:

$$
L_{\mathrm{collapse}} \times \phi_{\mathrm{refine}}
$$

- If collapse exists, refine when collapse index is late in horizon:

$$
s_{\mathrm{collapse}}\ge s_{\max}\times\phi_{\mathrm{refine}}
$$

Refinement reruns curvature with:

$$
\max(b_{\mathrm{curv}}+1,\;b_{\mathrm{adapt}})
$$

where:

- `L_collapse` is `collapse_loss_threshold`.
- `phi_refine` is `adaptive_curvature_trigger_fraction`.
- `s_collapse` is `collapse_step`.
- `s_max` is `max_steps`.
- `b_curv` is `curvature.max_batches`.
- `b_adapt` is `adaptive_curvature_max_batches`.

## 9. Trust-Region Warning Logic

A local-validity warning is appended when:

\[
\frac{\max_{t}\,\operatorname{drift}(t)}
     {\max\!\left(\lVert \Delta\theta_0 \rVert_2,\,10^{-12}\right)}
\ge r_{\mathrm{trust}}
\]

where $r_{\mathrm{trust}}$ is `trust_region_warning_ratio`.

## 10. Skipped LoRA Report (Non-Strict Mode)

If mode is `lora` and no matching trainable LoRA parameters with
`require_lora_match=False`, the pipeline returns:

- empty subspace;
- zeroed risk/curvature/forecast arrays;
- `collapse_step=None` and `collapse_time=None`;
- warning that analysis was skipped.

## 11. AlignGuard-Style Mitigation Objective

Given current selected LoRA parameter vector $\theta$ and reference vector
$\theta_{\text{ref}}$:

$$
\Delta = \theta - \theta_{\text{ref}}
$$

Decompose by Fisher subspace basis $B$:

$$
\Delta_A = BB^\top \Delta,\qquad
\Delta_T = \Delta - \Delta_A
$$

### 11.1 Alignment penalty

$$
\mathcal{L}_A =
\lambda_a \sum_{i=1}^{d}\lambda_i \langle b_i,\Delta_A\rangle^2
$$

### 11.2 Task-stability penalty

Choose diagonal metric $h$ by `h_type`:

- identity: $h_j=1$,
- fisher diagonal: $h_j = \max(\operatorname{diag}(F)_j, 0)$,
- custom: user-provided nonnegative diagonal.

$$
\mathcal{L}_T =
\lambda_t \sum_{j=1}^{n} h_j (\Delta_T)_j^2
$$

### 11.3 Collision penalties

Riemannian overlap term:

$$
\eta_j = 1+\beta\,\sigma\left(|(\Delta_A+\Delta_T)_j|-\tau\right)
$$

$$
\mathcal{C}_{\text{rm}}=
\operatorname{mean}_j\left[\eta_j\,|(\Delta_A)_j(\Delta_T)_j|\right]
$$

Geodesic overlap term:

$$
\mathcal{C}_{\text{geo}}=
\frac{(\Delta_A^\top \Delta_T)^2}
{\|\Delta_A\|_2^2\|\Delta_T\|_2^2+\epsilon}
$$

Blended collision penalty:

$$
\mathcal{L}_{\text{coll}}=
\lambda_{nc}\left(
\alpha \mathcal{C}_{\text{rm}} + (1-\alpha)\mathcal{C}_{\text{geo}}
\right)
$$

Total regularized objective:

$$
\mathcal{L}_{\text{total}}=
\mathcal{L}_{\text{task}}+\mathcal{L}_A+\mathcal{L}_T+\mathcal{L}_{\text{coll}}
$$

## 12. CLI/Demo Logic

- CLI exposes `demo` command with mode in `{full, lora}`.
- Demo builds a small synthetic model/data setup and runs full pipeline.
- Output artifacts:
  - module sensitivity bar chart,
  - forecast curve with collapse marker when present.

## 13. Output Contracts

`RiskAssessmentReport` fields:

- `subspace`: slices, Fisher eigenpairs, diagonal, module scores, top weights.
- `initial_risk`: overlap metrics and curvature-trigger flag.
- `curvature`: $\hat{\gamma}$, $\hat{\epsilon}$, acceleration norms.
- `forecast`: arrays for steps, time, drift, quartic term, estimated loss, collapse indices.
- `warning`: human-readable decision summary.

## 14. Scope and Approximation Notes

- Uses empirical Fisher from finite sample gradients, not exact Fisher.
- Forecast is a practical lower-bound style approximation with configurable
  thresholds/horizons.
- Auto backend may deliberately fall back to loop when parity probes fail.
- Mixed precision is handled by explicit promotion/alignment where spectral and
  projection computations require consistent dtypes.
