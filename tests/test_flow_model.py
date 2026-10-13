import numpy as np
import torch

from models.cap_flow import (CapFlow, cap_flow_matching_loss, integrate_cap_flow,
                             matched_residual_prediction, matched_residual_loss)


def _inputs(n=6, p=3, r=8, l=4, k=2):
    g = torch.Generator().manual_seed(0)
    return dict(
        x0=torch.randn(n, p, generator=g),
        x1=torch.randn(n, p, generator=g),
        query=torch.randn(n, r, generator=g),
        ref=torch.randn(n, r, generator=g),
        retz=torch.randn(n, k, r, generator=g),
        spatial=torch.randn(n, l, r, generator=g),
        disp=torch.randn(n, l, 2, generator=g),
        sim=torch.randn(n, 1, generator=g),
    )


def test_fixed_source_is_batch_composition_invariant():
    model = CapFlow(protein_dim=3, rna_dim=8, hidden_dim=32, depth=1, n_heads=4, n_near=2, k_retrieved=2)
    model.prior_adapter.set_retrieval_mean(np.array([0.1, -0.2, 0.3], np.float32))
    model.prior_adapter.set_de_shrink(np.array([1.2, 0.9, 1.1], np.float32))
    model.prior_adapter.set_logvar_bias(np.log(np.array([0.2, 0.3, 0.4], np.float32)))
    d = _inputs(n=4)
    mu_all, _ = model.prior_adapter(d["x0"], d["query"], d["spatial"], d["sim"])
    mu_one, _ = model.prior_adapter(d["x0"][:1], d["query"][:1], d["spatial"][:1], d["sim"][:1])
    assert torch.allclose(mu_all[:1], mu_one, atol=1e-6)


def test_flow_matching_and_eight_step_inference_smoke():
    torch.manual_seed(0)
    model = CapFlow(protein_dim=3, rna_dim=8, hidden_dim=32, depth=1, n_heads=4, n_near=2, k_retrieved=2)
    model.prior_adapter.set_retrieval_mean(np.zeros(3, np.float32))
    model.prior_adapter.set_de_shrink(np.ones(3, np.float32))
    model.prior_adapter.set_logvar_bias(np.log(np.full(3, 0.2, np.float32)))
    d = _inputs()
    out = cap_flow_matching_loss(
        model, d["x0"], d["x1"], d["query"], d["ref"], d["retz"], d["spatial"], d["disp"], d["sim"],
        flow_path="straight",
    )
    assert torch.isfinite(out.loss)

    calls = {"n": 0}
    original = model.forward
    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)
    model.forward = counted
    pred = integrate_cap_flow(
        model, d["x0"], d["query"], d["ref"], d["retz"], d["spatial"], d["disp"], d["sim"], nfe=8,
    )
    assert pred.shape == d["x0"].shape
    assert calls["n"] == 8


def test_matched_residual_is_direct_and_calibration_is_fixed():
    model = CapFlow(protein_dim=3, rna_dim=8, hidden_dim=32, depth=1, n_heads=4,
                    n_near=2, k_retrieved=2, dropout=0.)
    model.prior_adapter.set_retrieval_mean(np.array([.1,.2,.3],np.float32))
    model.prior_adapter.set_de_shrink(np.array([.5,2.,4.],np.float32))
    d = _inputs()
    calls = []
    original = model.forward
    def counted(x, t, *args, **kwargs):
        calls.append(t.detach().clone())
        return original(x, t, *args, **kwargs)
    model.forward = counted
    prediction = matched_residual_prediction(model,d['x0'],d['query'],d['ref'],d['retz'],
                                            d['spatial'],d['disp'],d['sim'])
    mu,_ = model.prior_adapter(d['x0'],d['query'],d['spatial'],d['sim'])
    # Zero-initialized velocity head makes the initial direct output exactly mu.
    assert torch.allclose(prediction,mu)
    assert len(calls)==1 and torch.count_nonzero(calls[0])==0
    loss=matched_residual_loss(model,d['x0'],d['x1'],d['query'],d['ref'],d['retz'],
                              d['spatial'],d['disp'],d['sim'])
    loss.backward()
    assert model.velocity.velocity_head.weight.grad is not None
    assert list(model.prior_adapter.parameters())==[]
