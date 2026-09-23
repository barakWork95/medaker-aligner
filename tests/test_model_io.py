"""Model I/O without loading weights: the int8 skeleton must never hold float32 Linear weights."""
import torch
from torch.ao.nn.quantized.dynamic import Linear as QLinear

from app import model_io


def test_int8_skeleton_has_only_quantized_linears_and_small_fp32_footprint():
    model = model_io._int8_skeleton()
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    qlinears = [m for m in model.modules() if isinstance(m, QLinear)]
    assert not linears
    assert len(qlinears) == 146  # 24 layers × 6 + feature projection + aux head
    fp32 = sum(p.numel() for p in model.parameters())
    assert fp32 < 15_000_000  # only extractor / norms / positional conv / head remain float32
    assert qlinears[0].in_features > 1  # placeholders remember the real shape


def test_configure_torch_is_idempotent_and_single_threaded(monkeypatch):
    monkeypatch.setenv("TORCH_THREADS", "1")
    model_io._configure_torch()
    model_io._configure_torch()
    assert torch.get_num_threads() == 1
    assert not torch.is_grad_enabled()


def test_load_model_prefers_int8_then_falls_back(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(model_io, "load_int8", lambda p: calls.append(("int8", p)) or "M")
    monkeypatch.setattr(model_io, "INT8_URL", "")
    monkeypatch.setattr(model_io, "convert_to_int8", lambda out: calls.append(("convert", out)) or out)

    class FakeBundle:
        def get_model(self, with_star):
            calls.append(("fp32", with_star))

            class M:
                def eval(self):
                    return "F"

            return M()

    monkeypatch.setattr(model_io, "_bundle", lambda: FakeBundle())
    missing = tmp_path / "none.pt"
    assert model_io.load_model(prefer_int8=True, int8_path=missing) == ("F", "fp32")
    monkeypatch.setenv("MMS_FA_INT8_REQUIRED", "1")
    assert model_io.load_model(prefer_int8=True, int8_path=missing) == ("M", "int8")
    assert ("convert", missing) in calls
    present = tmp_path / "int8.pt"
    present.write_bytes(b"x")
    assert model_io.load_model(prefer_int8=True, int8_path=present) == ("M", "int8")
