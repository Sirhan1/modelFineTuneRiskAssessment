import torch

from alignment_risk.fisher import FisherConfig, FisherSubspaceAnalyzer


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _build_data() -> list[tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(123)
    x = torch.randn(12, 4)
    y = torch.randint(0, 2, (12,))
    return [(x, y)]


def _loss_fn(model: torch.nn.Module, batch: object) -> torch.Tensor:
    x, y = batch  # type: ignore[misc]
    logits = model(x)
    return torch.nn.functional.cross_entropy(logits, y)


def _new_model() -> torch.nn.Module:
    torch.manual_seed(777)
    return _TinyClassifier()


def test_vmap_and_loop_backends_produce_similar_fisher_diagonal() -> None:
    data = _build_data()

    loop_analyzer = FisherSubspaceAnalyzer(
        FisherConfig(
            gradient_collection="loop",
            subspace_method="svd",
            max_samples=12,
            top_rank=4,
            max_parameters=128,
        )
    )
    vmap_analyzer = FisherSubspaceAnalyzer(
        FisherConfig(
            gradient_collection="vmap",
            subspace_method="svd",
            max_samples=12,
            top_rank=4,
            max_parameters=128,
            vmap_chunk_size=6,
        )
    )

    loop_out = loop_analyzer.analyze(_new_model(), data, _loss_fn)
    vmap_out = vmap_analyzer.analyze(_new_model(), data, _loss_fn)

    assert torch.allclose(loop_out.fisher_diagonal, vmap_out.fisher_diagonal, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        loop_out.fisher_eigenvalues,
        vmap_out.fisher_eigenvalues,
        atol=1e-4,
        rtol=1e-4,
    )


def test_vmap_and_loop_backends_match_when_loss_reads_model_parameters() -> None:
    data = _build_data()

    def _loss_with_parameter_regularization(model: torch.nn.Module, batch: object) -> torch.Tensor:
        x, y = batch  # type: ignore[misc]
        logits = model(x)
        ce = torch.nn.functional.cross_entropy(logits, y)
        l2 = torch.zeros((), dtype=logits.dtype, device=logits.device)
        for p in model.parameters():
            l2 = l2 + (p**2).sum()
        return ce + 0.1 * l2

    loop_analyzer = FisherSubspaceAnalyzer(
        FisherConfig(
            gradient_collection="loop",
            subspace_method="svd",
            max_samples=12,
            top_rank=4,
            max_parameters=128,
        )
    )
    vmap_analyzer = FisherSubspaceAnalyzer(
        FisherConfig(
            gradient_collection="vmap",
            subspace_method="svd",
            max_samples=12,
            top_rank=4,
            max_parameters=128,
            vmap_chunk_size=6,
        )
    )

    loop_out = loop_analyzer.analyze(_new_model(), data, _loss_with_parameter_regularization)
    vmap_out = vmap_analyzer.analyze(_new_model(), data, _loss_with_parameter_regularization)

    assert torch.allclose(loop_out.fisher_diagonal, vmap_out.fisher_diagonal, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        loop_out.fisher_eigenvalues,
        vmap_out.fisher_eigenvalues,
        atol=1e-4,
        rtol=1e-4,
    )


def test_vmap_backend_handles_scalar_tensor_metadata_in_dict_batches() -> None:
    torch.manual_seed(11)
    x = torch.randn(10, 4)
    y = torch.randint(0, 2, (10,))
    data = [
        {
            "meta_id": torch.tensor(0),
            "x": x,
            "y": y,
        }
    ]

    def _loss_fn_dict(model: torch.nn.Module, batch: object) -> torch.Tensor:
        b = batch  # type: ignore[assignment]
        logits = model(b["x"])  # type: ignore[index]
        return torch.nn.functional.cross_entropy(logits, b["y"])  # type: ignore[index]

    out = FisherSubspaceAnalyzer(
        FisherConfig(
            gradient_collection="vmap",
            subspace_method="svd",
            max_samples=10,
            top_rank=3,
            max_parameters=128,
        )
    ).analyze(_new_model(), data, _loss_fn_dict)

    assert out.fisher_diagonal.numel() > 0


def test_fisher_subspace_method_options_return_expected_shapes() -> None:
    data = _build_data()
    cfg_base = dict(
        gradient_collection="loop",
        max_samples=12,
        top_rank=3,
        max_parameters=128,
    )

    svd_out = FisherSubspaceAnalyzer(FisherConfig(subspace_method="svd", **cfg_base)).analyze(
        _new_model(),
        data,
        _loss_fn,
    )
    rnd_out = FisherSubspaceAnalyzer(
        FisherConfig(
            subspace_method="randomized_svd",
            randomized_oversample=2,
            randomized_niter=1,
            **cfg_base,
        )
    ).analyze(_new_model(), data, _loss_fn)
    diag_out = FisherSubspaceAnalyzer(FisherConfig(subspace_method="diag_topk", **cfg_base)).analyze(
        _new_model(),
        data,
        _loss_fn,
    )

    n_params = svd_out.fisher_diagonal.numel()
    assert svd_out.fisher_eigenvectors.shape == (n_params, 3)
    assert rnd_out.fisher_eigenvectors.shape == (n_params, 3)
    assert diag_out.fisher_eigenvectors.shape == (n_params, 3)

    # diagonal-topk method should produce canonical one-hot basis columns.
    assert torch.all(diag_out.fisher_eigenvectors.sum(dim=0) == 1.0)
    assert torch.all((diag_out.fisher_eigenvectors == 0.0) | (diag_out.fisher_eigenvectors == 1.0))


def test_fisher_rank_can_be_selected_by_explained_variance() -> None:
    class _SyntheticGradAnalyzer(FisherSubspaceAnalyzer):
        def _collect_per_sample_gradients(  # type: ignore[override]
            self,
            model: torch.nn.Module,
            parameter_names: list[str],
            params: list[torch.nn.Parameter],
            dataloader: object,
            loss_fn: object,
            max_samples: int,
            device: torch.device,
        ) -> list[torch.Tensor]:
            # Columns are orthogonal; singular values are [3, 2, 1].
            g = torch.tensor(
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            return [g[i] for i in range(g.shape[0])]

    model = torch.nn.Linear(3, 1, bias=False)
    cfg = FisherConfig(
        gradient_collection="loop",
        subspace_method="svd",
        top_rank=3,
        target_explained_variance=0.9,
        max_samples=3,
        max_parameters=32,
    )
    out = _SyntheticGradAnalyzer(cfg).analyze(model, dataloader=[], loss_fn=lambda m, b: torch.tensor(0.0))
    # Eigenvalues proportional to [9, 4, 1], so first two explain > 0.9.
    assert out.fisher_eigenvalues.numel() == 2


def test_module_scores_are_aggregated_by_module_name() -> None:
    data = _build_data()
    out = FisherSubspaceAnalyzer(
        FisherConfig(
            gradient_collection="loop",
            subspace_method="svd",
            max_samples=12,
            top_rank=3,
            max_parameters=128,
        )
    ).analyze(_new_model(), data, _loss_fn)

    assert "linear" in out.module_scores
    assert "linear.weight" not in out.module_scores
    assert "linear.bias" not in out.module_scores
