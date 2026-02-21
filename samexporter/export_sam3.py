import argparse
import pathlib

import numpy as np
import onnx
import torch
from onnxsim import simplify
from torchvision.transforms import v2

# Mock triton for Windows
import sys
from unittest.mock import MagicMock
mock_triton = MagicMock()
sys.modules["triton"] = mock_triton
sys.modules["triton.language"] = MagicMock()
sys.modules["torch._inductor.runtime.triton_helpers"] = MagicMock()

# Ensure sam3 is in PYTHONPATH
import os
# This file is at samexporter/samexporter/export_sam3.py
samexporter_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Submodule is at samexporter/sam3.
# Package 'sam3' is at samexporter/sam3/sam3.
# So we add samexporter/sam3 to sys.path.
sys.path.append(os.path.join(samexporter_root, "sam3"))
sys.path.append(samexporter_root)

try:
    from sam3.model.sam3_image import Sam3Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from osam._models.yoloworld.clip import tokenize
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure the sam3 submodule inside samexporter is available.")
    Sam3Image = object
    Sam3Processor = object
    build_sam3_image_model = lambda: None
    tokenize = lambda x: None


def get_replace_freqs_cis(module: torch.nn.Module) -> None:
    """Replace complex freqs_cis buffers with separate real/imag float buffers.

    ONNX does not support complex-valued tensors, so the complex RoPE
    (rotary positional embedding) buffer must be split into its real
    (cosine) and imaginary (sine) components before export.
    """
    if hasattr(module, "freqs_cis"):
        freqs_cos = module.freqs_cis.real.float()
        freqs_sin = module.freqs_cis.imag.float()
        module.register_buffer("freqs_cos", freqs_cos)
        module.register_buffer("freqs_sin", freqs_sin)
        del module.freqs_cis
    for child in module.children():
        get_replace_freqs_cis(child)


class SAM3ImageEncoder(torch.nn.Module):
    """Wraps the SAM3 image backbone for ONNX export.

    Input:  image  – uint8 tensor of shape (3, 1008, 1008) in RGB order.
    Output: 6 float tensors – vision_pos_enc_{0,1,2} and backbone_fpn_{0,1,2}.

    Normalization (mean=0.5, std=0.5 per channel, i.e. mapping [0,255]→[−1,1])
    is baked into the ONNX graph so that inference tools do not need to
    pre-process the image themselves.
    """

    def __init__(self, processor: Sam3Processor) -> None:
        super().__init__()
        self._processor: Sam3Processor = processor
        # Normalise uint8 [0,255] to float [-1, 1] – identical to the
        # reference sam3-onnx export (export_onnx.py).
        self._transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),   # uint8 → float [0,1]
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1,1]
            ]
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # image: (3, H, W) uint8 → normalise → (1, 3, H, W) float
        image = self._transform(image).unsqueeze(0)
        backbone_out = self._processor.model.backbone._forward_image_no_act_ckpt(image)
        # Remove keys that are not needed by the decoder and would add
        # unnecessary overhead to the ONNX graph.
        backbone_out.pop("vision_features", None)
        backbone_out.pop("sam2_backbone_out", None)
        assert len(backbone_out["vision_pos_enc"]) == 3
        assert len(backbone_out["backbone_fpn"]) == 3
        return *backbone_out["vision_pos_enc"], *backbone_out["backbone_fpn"]


class SAM3LanguageEncoder(torch.nn.Module):
    def __init__(self, processor: Sam3Processor) -> None:
        super().__init__()
        self._processor: Sam3Processor = processor

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model: Sam3Image = self._processor.model

        # VETextEncoder forward pass
        text_attention_mask = (tokens != 0).bool()
        inputs_embeds = model.backbone.language_backbone.encoder.token_embedding(tokens)
        _, text_memory = model.backbone.language_backbone.encoder(tokens)

        assert text_memory.shape[1] == inputs_embeds.shape[1]
        text_attention_mask = text_attention_mask.ne(1)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = model.backbone.language_backbone.resizer(text_memory)
        # Output names (set at export time):
        #   text_attention_mask – bool   [1, seq_len]
        #   text_memory         – float  [seq_len, 1, 256]
        #   text_embeds         – float  [seq_len, 1, 1024]
        return text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1)


class SAM3Decoder(torch.nn.Module):
    def __init__(self, model: Sam3Image, processor: Sam3Processor) -> None:
        super().__init__()
        self._model = model
        self._processor = processor

    def forward(
        self,
        original_height: torch.Tensor,
        original_width: torch.Tensor,
        vision_pos_enc_0: torch.Tensor,
        vision_pos_enc_1: torch.Tensor,
        vision_pos_enc_2: torch.Tensor,
        backbone_fpn_0: torch.Tensor,
        backbone_fpn_1: torch.Tensor,
        backbone_fpn_2: torch.Tensor,
        language_mask: torch.Tensor,
        language_features: torch.Tensor,
        language_embeds: torch.Tensor,
        box_coords: torch.Tensor,
        box_labels: torch.Tensor,
        box_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geometric_prompt = self._model._get_dummy_prompt()
        geometric_prompt.box_embeddings = box_coords
        geometric_prompt.box_labels = box_labels
        geometric_prompt.box_mask = box_masks
        state = {
            "original_height": original_height,
            "original_width": original_width,
            "backbone_out": {
                "vision_pos_enc": [
                    vision_pos_enc_0,
                    vision_pos_enc_1,
                    vision_pos_enc_2,
                ],
                "backbone_fpn": [
                    backbone_fpn_0,
                    backbone_fpn_1,
                    backbone_fpn_2,
                ],
                "language_mask": language_mask,
                "language_features": language_features,
                "language_embeds": language_embeds,
            },
            "geometric_prompt": geometric_prompt,
        }
        result = self._processor._forward_grounding(state)
        return result["boxes"], result["scores"], result["masks"]


def export_sam3(output_dir: str, opset: int = 18, simplify_model: bool = False):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model()
    # Replace complex RoPE buffers with float cos/sin – required for ONNX.
    get_replace_freqs_cis(model)
    processor = Sam3Processor(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ── Image Encoder ────────────────────────────────────────────────────────
    print("Exporting Image Encoder...")
    image_encoder = SAM3ImageEncoder(processor)
    # Input: uint8 (3, 1008, 1008) – normalization is baked into the model.
    dummy_image = torch.zeros(3, 1008, 1008, dtype=torch.uint8).to(device)
    encoder_path = output_dir / "sam3_image_encoder.onnx"
    torch.onnx.utils.export(
        image_encoder,
        args=(dummy_image,),
        f=str(encoder_path),
        export_params=True,
        input_names=["image"],
        output_names=[
            "vision_pos_enc_0",
            "vision_pos_enc_1",
            "vision_pos_enc_2",
            "backbone_fpn_0",
            "backbone_fpn_1",
            "backbone_fpn_2",
        ],
        opset_version=opset,
    )
    print(f"Saved Image Encoder to {encoder_path}")

    # ── Language Encoder ─────────────────────────────────────────────────────
    print("Exporting Language Encoder...")
    language_encoder = SAM3LanguageEncoder(processor)
    dummy_tokens = torch.zeros(1, 32, dtype=torch.long).to(device)
    language_path = output_dir / "sam3_language_encoder.onnx"
    torch.onnx.utils.export(
        language_encoder,
        args=(dummy_tokens,),
        f=str(language_path),
        export_params=True,
        input_names=["tokens"],
        # Names match the actual ONNX model output names used by inference code.
        output_names=["text_attention_mask", "text_memory", "text_embeds"],
        opset_version=opset,
    )
    print(f"Saved Language Encoder to {language_path}")

    # Get dummy feature tensors for decoder export
    with torch.no_grad():
        vpe0, vpe1, vpe2, fpn0, fpn1, fpn2 = image_encoder(dummy_image)
        l_mask, l_feat, l_embed = language_encoder(dummy_tokens)

    # ── Decoder ──────────────────────────────────────────────────────────────
    print("Exporting Decoder...")
    decoder = SAM3Decoder(model, processor)
    decoder_path = output_dir / "sam3_decoder.onnx"

    box_coords = torch.zeros(1, 1, 4).to(device)
    box_labels = torch.ones(1, 1, dtype=torch.long).to(device)
    # box_masks=True means "no real box" (dummy / masked out).
    box_masks = torch.ones(1, 1, dtype=torch.bool).to(device)
    orig_h = torch.tensor(1008).to(device)
    orig_w = torch.tensor(1008).to(device)

    torch.onnx.utils.export(
        decoder,
        args=(
            orig_h, orig_w,
            vpe0, vpe1, vpe2,
            fpn0, fpn1, fpn2,
            l_mask, l_feat, l_embed,
            box_coords, box_labels, box_masks,
        ),
        f=str(decoder_path),
        export_params=True,
        input_names=[
            "original_height",
            "original_width",
            "vision_pos_enc_0",
            "vision_pos_enc_1",
            "vision_pos_enc_2",
            "backbone_fpn_0",
            "backbone_fpn_1",
            "backbone_fpn_2",
            "language_mask",
            "language_features",
            "language_embeds",
            "box_coords",
            "box_labels",
            "box_masks",
        ],
        output_names=["boxes", "scores", "masks"],
        opset_version=opset,
    )
    print(f"Saved Decoder to {decoder_path}")

    # ── Simplify models conditionally ─────────────────────────────────────────
    if simplify_model:
        for path in [encoder_path, language_path, decoder_path]:
            print(f"Simplifying {path}...")
            try:
                onnx_model = onnx.load(str(path))
                model_simp, check = simplify(onnx_model)
                assert check, "Simplified ONNX model could not be validated"
                onnx.save(model_simp, str(path))
                print(f"  → simplified OK")
            except Exception as e:
                print(f"  → simplification failed ({e}), keeping original")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SAM3 to ONNX")
    parser.add_argument(
        "--output_dir", type=str, default="output_models/sam3",
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--opset", type=int, default=18, help="ONNX opset version"
    )
    parser.add_argument(
        "--simplify", action="store_true", help="Simplify ONNX models"
    )
    args = parser.parse_args()
    export_sam3(args.output_dir, args.opset, args.simplify)
