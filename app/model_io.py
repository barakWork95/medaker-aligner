"""
Memory-efficient MMS_FA model loading.

torchaudio's `MMS_FA.get_model()` reads the 1.2 GB float32 checkpoint into RAM and builds the
float32 model (≈ 1.9 GB resident, ≈ 2.6 GB peak). That cannot run on a 512 MB instance, and
neither can quantising at start-up (the float32 skeleton alone is 1.2 GB).

So the model is converted ONCE (build time / first run on a big machine) with
`convert_to_int8()` — dynamic int8 quantisation of every Linear layer — and saved as a pickled
module. `load_int8()` then rebuilds it straight from the int8 packed weights, memory-mapped,
never materialising float32 weights. Everything else (extractor, layer norms, positional conv)
stays float32: ~15 M parameters.

Accuracy: dynamic int8 on the transformer's Linear layers changes CTC posteriors marginally;
alignment paths are unaffected in practice (see tests/test_align_e2e.py which runs on whatever
variant is active).
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Optional

DEFAULT_INT8_PATH = Path(os.environ.get("MMS_FA_INT8", os.path.join(os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")), "mms_fa_int8.pt")))
# Optional: URL of a prebuilt int8 artifact (e.g. a GitHub release asset) downloaded on first start.
INT8_URL = os.environ.get("MMS_FA_INT8_URL", "")


def _configure_torch() -> None:
    import torch

    threads = int(os.environ.get("TORCH_THREADS", "1"))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set / parallel work started
    torch.set_grad_enabled(False)
    if torch.backends.quantized.engine == "none":
        for engine in ("fbgemm", "qnnpack"):
            if engine in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = engine
                break


def _bundle():
    import torchaudio

    return torchaudio.pipelines.MMS_FA


def _int8_skeleton():
    """
    The MMS_FA module tree with every Linear replaced by a PLACEHOLDER int8 dynamic Linear
    (1×1 packed weight), built WITHOUT allocating float32 weights: the tree is created on the
    meta device, the placeholders are inserted, and the remaining (12.7 M) float32 tensors are
    materialised empty. Real packed weights are streamed in by load_int8().
    """
    import torch
    from torch.ao.nn.quantized.dynamic import Linear as QLinear
    from torchaudio.pipelines._wav2vec2 import utils

    _configure_torch()  # selects a quantized engine (fbgemm/x86 on Linux, qnnpack on macOS)
    bundle = _bundle()
    with torch.device("meta"):
        core = utils._get_model(bundle._model_type, bundle._params)
        model = utils._extend_model(core, normalize_waveform=bundle._normalize_waveform, apply_log_softmax=True, append_star=False)
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear):
                q = QLinear(1, 1, bias_=child.bias is not None, dtype=torch.qint8)
                q.in_features, q.out_features = child.in_features, child.out_features
                setattr(module, child_name, q)
    model.to_empty(device="cpu")
    return model


def _fp32_via_mmap():
    """
    Float32 model with the checkpoint memory-mapped instead of read into RAM: peak ≈ 1.3 GB
    instead of the bundle's ≈ 2.6 GB (matters for Docker builds on small builders).
    """
    import torch
    from torchaudio.pipelines._wav2vec2 import utils

    bundle = _bundle()
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(bundle._path).name
    if not path.exists():
        torch.hub.download_url_to_file(f"https://download.pytorch.org/torchaudio/models/{bundle._path}", str(path), progress=False)
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    utils._remove_aux_axes(sd, bundle._remove_aux_axis)
    with torch.device("meta"):
        core = utils._get_model(bundle._model_type, bundle._params)
    core.to_empty(device="cpu")
    core.load_state_dict(sd)
    del sd
    model = utils._extend_model(core, normalize_waveform=bundle._normalize_waveform, apply_log_softmax=True, append_star=False)
    return model.eval()


def convert_to_int8(out: Path = DEFAULT_INT8_PATH) -> Path:
    """Float32 checkpoint → int8 dynamic-quantised state dict on disk (peak ≈ 1.7 GB, once)."""
    import torch

    _configure_torch()
    model = _fp32_via_mmap()
    q = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    del model
    gc.collect()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(q.state_dict(), out)
    del q
    gc.collect()
    return out


def load_int8(path: Path = DEFAULT_INT8_PATH):
    """
    Rebuild the int8 model from its state dict with the smallest possible peak:
      - the file is memory-mapped (pages are file-backed and dropped once consumed),
      - each Linear's packed weight is set directly from the file, one layer at a time,
        and its entry deleted, so no second copy of the 300 MB of weights ever exists,
      - the 12.7 M float32 tensors are loaded last.
    """
    import torch
    from torch.ao.nn.quantized.dynamic import Linear as QLinear

    _configure_torch()
    model = _int8_skeleton()
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    modules = dict(model.named_modules())
    for key in [k for k in list(sd.keys()) if k.endswith("._packed_params._packed_params")]:
        prefix = key[: -len("._packed_params._packed_params")]
        module = modules.get(prefix)
        if not isinstance(module, QLinear):
            raise RuntimeError(f"int8 artifact/skeleton mismatch at {prefix}")
        weight, bias = sd.pop(key)
        module.set_weight_bias(weight, bias)
        sd.pop(prefix + "._packed_params.dtype", None)
        del weight, bias
    # float32 leftovers (feature extractor, norms, positional conv, output head): copy by name.
    # Skeleton tensors start uninitialised, so every one of them must be written.
    targets = dict(model.named_parameters())
    targets.update(model.named_buffers())
    written: set[str] = set()
    unexpected: list[str] = []
    with torch.no_grad():
        for key in list(sd.keys()):
            target = targets.get(key)
            if target is None:
                # quantized Linear layers also save scale / zero_point; dynamic quantisation
                # re-derives activation scales per call, so these are not needed
                if key.endswith((".scale", ".zero_point")):
                    sd.pop(key)
                    continue
                unexpected.append(key)
                continue
            target.copy_(sd.pop(key))
            written.add(key)
    del sd
    gc.collect()
    if unexpected:
        raise RuntimeError(f"int8 artifact has unexpected tensors: {unexpected[:3]}")
    not_written = sorted(k for k in targets if k not in written and "_packed_params" not in k)
    if not_written:
        raise RuntimeError(f"int8 artifact incomplete, uninitialised tensors: {not_written[:3]}")
    model.eval()
    return model


def fetch_int8(url: str, out: Path = DEFAULT_INT8_PATH) -> Path:
    """Stream a prebuilt int8 artifact to disk (no large buffers in memory)."""
    import urllib.request

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(out)
    return out


def load_model(prefer_int8: bool = True, int8_path: Path = DEFAULT_INT8_PATH):
    """
    int8 artifact if present → else downloaded from MMS_FA_INT8_URL → else converted when
    MMS_FA_INT8_REQUIRED=1 → else the float32 bundle. Returns (model, variant).
    """
    _configure_torch()
    if prefer_int8:
        if not int8_path.exists() and INT8_URL:
            fetch_int8(INT8_URL, int8_path)
        if int8_path.exists():
            return load_int8(int8_path), "int8"
        if os.environ.get("MMS_FA_INT8_REQUIRED") == "1":
            convert_to_int8(int8_path)
            return load_int8(int8_path), "int8"
    model = _bundle().get_model(with_star=False).eval()
    gc.collect()
    return model, "fp32"


def peak_rss_mb() -> Optional[float]:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (2**20 if os.uname().sysname == "Darwin" else 2**10), 1)
    except Exception:  # pragma: no cover
        return None
