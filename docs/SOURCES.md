# Source Mapping

This repository's methods are grounded in two papers (citation keys follow
`docs/CITATIONS.md`):

1. **[AIC-2026]** Max Springer et al., *The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety* (arXiv:2602.15799v1, 2026-02-17).
   - PDF: `https://arxiv.org/pdf/2602.15799`
   - Abstract page: `https://arxiv.org/abs/2602.15799`
2. **[ALIGNGUARD-2025]** Amitava Das et al., *AlignGuard-LoRA: Alignment-Preserving Fine-Tuning via Fisher-Guided Decomposition and Riemannian-Geodesic Collision Regularization* (arXiv:2508.02079v1, 2025-08-04).
   - PDF: `https://arxiv.org/pdf/2508.02079`
   - Abstract page: `https://arxiv.org/abs/2508.02079`

This is an implementation-oriented toolkit, not a line-by-line reproduction of either paper. The links below are the closest source sections for each module.
For full equations and concrete runtime decision logic, see `docs/MATH.md`.

## Module to source mapping

- `src/alignment_risk/fisher.py`
  - Uses empirical Fisher geometry and low-rank sensitivity subspaces.
  - [AIC-2026] Section 3.2-3.3, Proposition 3.3 and Definition 3.4 (page 5).
  - [ALIGNGUARD-2025] Section 4.1 (page 4), plus Appendix B Fisher estimation notes (pages 45-46).

- `src/alignment_risk/orthogonality.py`
  - Uses first-order projection/overlap against the sensitive subspace to flag whether curvature checks are needed.
  - [AIC-2026] Initial Orthogonality condition in Definition 5.1 (page 8).

- `src/alignment_risk/curvature.py`
  - Estimates curvature coupling via a directional second-order term (`H g` style) and projects it onto the Fisher subspace.
  - [AIC-2026] Curvature Coupling discussion and AIC condition 3 in Section 5.2-5.3 (page 8), and Theorem 6.2 drift term (page 9).

- `src/alignment_risk/forecast.py`
  - Encodes a practical lower-bound-style drift and quartic degradation forecast.
  - [AIC-2026] Theorem 6.2 and Corollary 6.3 (pages 9-10), plus informal quartic summary (page 3).

- `src/alignment_risk/mitigation.py`
  - Implements AlignGuard-style decomposition and regularization:
    - `DeltaW = DeltaW_A + DeltaW_T`
    - Fisher-weighted alignment penalty
    - task-stability penalty on the orthogonal component
    - blended collision penalties (Riemannian + geodesic)
  - AlignGuard source:
    - Main decomposition/regularization framing: Section 4.1-4.2 and objective figure (pages 4-6).
    - Collision energies and blend details: Appendix C (pages 45-46).
  - Supplemental formula recap: FAQ appendix pages 20-25 and 30-32.

- `src/alignment_risk/pipeline.py`
  - Orchestrates the same sequence as the AIC narrative:
    1. low-rank sensitivity extraction,
    2. initial overlap check,
    3. curvature-induced drift estimate,
    4. quartic-style warning forecast.
  - [AIC-2026] Definition 5.1 and Section 6 (pages 8-10).
  - [ALIGNGUARD-2025] Section 4 and objective (pages 4-6).

- `src/alignment_risk/visualization.py`
  - Produces module-level Fisher sensitivity plots and forecast curves.
  - Source rationale: diagnostic visualization of Fisher spectra/overlap in both papers, especially [AIC-2026] Section 7 and [ALIGNGUARD-2025] Appendix B/C figures.

## Notes on interpretation

- [AIC-2026] is primarily theoretical; this repo uses finite-step and finite-data approximations for engineering use.
- [ALIGNGUARD-2025] introduces multiple objective terms; this repo implements a compact variant for LoRA regularization in `mitigation.py`.
- Forecast constants and thresholds in this repo are configurable heuristics (`ForecastConfig`) and should be calibrated per model/task.
