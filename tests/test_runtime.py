from unittest.mock import patch

import pytest

from samexporter.runtime import get_onnx_providers


@patch("onnxruntime.get_available_providers")
def test_auto_provider_order_includes_accelerators_and_cpu(mock_available):
    mock_available.return_value = [
        "CPUExecutionProvider",
        "AzureExecutionProvider",
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
    ]
    assert get_onnx_providers() == [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    ]


@patch("onnxruntime.get_available_providers")
def test_provider_aliases_and_cpu_fallback(mock_available):
    mock_available.return_value = [
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert get_onnx_providers("openvino,cuda") == [
        "OpenVINOExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


@patch("onnxruntime.get_available_providers")
def test_unavailable_explicit_provider_has_actionable_error(mock_available):
    mock_available.return_value = ["CPUExecutionProvider"]
    with pytest.raises(ValueError, match="CUDAExecutionProvider.*Available"):
        get_onnx_providers("cuda")
