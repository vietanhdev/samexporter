import torch

from samexporter.export_sam3 import (
    prepare_fused_mlps_for_onnx,
    prepare_rope_buffers_for_onnx,
)


class RopeModule(torch.nn.Module):
    def __init__(self, *, prepared: bool):
        super().__init__()
        self.use_rope_real = True
        self.register_buffer("freqs_cis", torch.ones(2, dtype=torch.complex64))
        if prepared:
            self.register_buffer("freqs_cis_real", self.freqs_cis.real)
            self.register_buffer("freqs_cis_imag", self.freqs_cis.imag)


def test_rope_preparation_keeps_required_complex_buffer():
    module = RopeModule(prepared=False)
    prepare_rope_buffers_for_onnx(module)
    assert hasattr(module, "freqs_cis")
    assert hasattr(module, "freqs_cis_real")
    assert hasattr(module, "freqs_cis_imag")
    assert module.use_rope_real


def test_rope_preparation_is_idempotent_for_current_sam3():
    module = RopeModule(prepared=True)
    real_buffer = module.freqs_cis_real
    prepare_rope_buffers_for_onnx(module)
    assert module.freqs_cis_real is real_buffer


def test_fused_mlp_preparation_ignores_unrelated_modules():
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    original_forward = module[0].forward
    prepare_fused_mlps_for_onnx(module)
    assert module[0].forward == original_forward
