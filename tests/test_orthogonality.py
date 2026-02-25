import torch

from alignment_risk.orthogonality import initial_risk_from_update
from alignment_risk.types import ParameterSlice, SensitivitySubspace


def _dummy_subspace() -> SensitivitySubspace:
    basis = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    return SensitivitySubspace(
        parameter_slices=[ParameterSlice(name="w", start=0, end=3, shape=(3,))],
        fisher_eigenvalues=torch.tensor([2.0, 1.0]),
        fisher_eigenvectors=basis,
        fisher_diagonal=torch.tensor([2.0, 1.0, 0.1]),
        module_scores={"w": 1.0333},
        top_weight_indices=torch.tensor([0, 1]),
        top_weight_scores=torch.tensor([2.0, 1.0]),
    )


def test_initial_risk_ratio() -> None:
    subspace = _dummy_subspace()
    update = torch.tensor([1.0, 0.0, 1.0])
    out = initial_risk_from_update(update, subspace, orthogonality_threshold=0.8)
    assert 0.0 < out.projected_ratio < 1.0
    assert out.trigger_curvature_check


def test_initial_risk_handles_mixed_update_and_basis_dtypes() -> None:
    subspace = _dummy_subspace()
    update = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float16)
    out = initial_risk_from_update(update, subspace, orthogonality_threshold=0.8)
    assert 0.0 <= out.projected_ratio <= 1.0
