import logging
import os
from collections.abc import Sequence

import onnxruntime

_PROVIDER_PREFERENCE = (
    "NvTensorRTRTXExecutionProvider",
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CANNExecutionProvider",
    "QNNExecutionProvider",
    "NnapiExecutionProvider",
    "VSINPUExecutionProvider",
    "WebNNExecutionProvider",
    "WebGpuExecutionProvider",
    "XnnpackExecutionProvider",
    "RknpuExecutionProvider",
    "VitisAIExecutionProvider",
    "ACLExecutionProvider",
    "ArmNNExecutionProvider",
    "DnnlExecutionProvider",
    "JsExecutionProvider",
    "AzureExecutionProvider",
    "CPUExecutionProvider",
)

_PROVIDER_ALIASES = {
    "tensorrt_rtx": "NvTensorRTRTXExecutionProvider",
    "trt_rtx": "NvTensorRTRTXExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cann": "CANNExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "nnapi": "NnapiExecutionProvider",
    "vsinpu": "VSINPUExecutionProvider",
    "webnn": "WebNNExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
    "rknpu": "RknpuExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "acl": "ACLExecutionProvider",
    "armnn": "ArmNNExecutionProvider",
    "dnnl": "DnnlExecutionProvider",
    "onednn": "DnnlExecutionProvider",
    "js": "JsExecutionProvider",
    "azure": "AzureExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def _parse_requested_providers(
    requested: str | Sequence[str] | None,
) -> list[str] | None:
    if requested is None:
        requested = os.environ.get("SAMEXPORTER_ONNX_PROVIDERS")
    if requested is None:
        return None
    values = requested.split(",") if isinstance(requested, str) else requested
    providers = []
    for value in values:
        name = value.strip()
        if not name:
            continue
        providers.append(_PROVIDER_ALIASES.get(name.lower(), name))
    return providers


def get_onnx_providers(
    requested: str | Sequence[str] | None = None,
) -> list[str]:
    """Return available ONNX Runtime providers in preference order.

    ``requested`` accepts exact ONNX Runtime provider names or short aliases
    such as ``tensorrt,cuda,cpu``. When omitted, the
    ``SAMEXPORTER_ONNX_PROVIDERS`` environment variable is honored, followed by
    automatic accelerator discovery. CPU is retained as the final fallback when
    it is installed.
    """
    available = onnxruntime.get_available_providers()
    requested_providers = _parse_requested_providers(requested)
    if requested_providers is not None:
        unavailable = [name for name in requested_providers if name not in available]
        if unavailable:
            raise ValueError(
                "Requested ONNX Runtime provider(s) are not installed: "
                + ", ".join(unavailable)
                + ". Available: "
                + ", ".join(available)
            )
        providers = list(dict.fromkeys(requested_providers))
    else:
        preference = [
            name
            for name in _PROVIDER_PREFERENCE
            if name in available and name != "CPUExecutionProvider"
        ]
        unknown = [name for name in available if name not in _PROVIDER_PREFERENCE]
        providers = preference + unknown

    if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    logging.info("ONNX Runtime providers: %s", ", ".join(providers))
    return providers
