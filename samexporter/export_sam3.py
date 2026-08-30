import argparse
import os
import pathlib
import sys
import types

import onnx
import torch
from torchvision.transforms import v2

from samexporter.upstream import prefer_pinned_upstream

# SAM3 can import optional Triton helpers on Windows, where Triton is normally
# unavailable. Never replace a real Linux Triton installation: doing so breaks
# PyTorch CUDA's lazy kernel registration.
if os.name == "nt":
    from importlib.machinery import ModuleSpec
    from unittest.mock import MagicMock

    mock_triton = MagicMock()
    mock_triton.__spec__ = ModuleSpec("triton", loader=None)
    sys.modules.setdefault("triton", mock_triton)
    sys.modules.setdefault("triton.language", MagicMock())
    sys.modules.setdefault("torch._inductor.runtime.triton_helpers", MagicMock())

# Ensure sam3 is in PYTHONPATH

# This file is at samexporter/samexporter/export_sam3.py
samexporter_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Submodule is at samexporter/sam3.
# Package 'sam3' is at samexporter/sam3/sam3.
# So we add samexporter/sam3 to sys.path.
prefer_pinned_upstream("sam3")
sys.path.append(samexporter_root)

try:
    from sam3.model.sam3_image import Sam3Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure the sam3 submodule inside samexporter is available.")
    Sam3Image = object
    Sam3Processor = object

    def build_sam3_image_model():
        return None


def prepare_rope_buffers_for_onnx(module: torch.nn.Module) -> None:
    """Replace complex freqs_cis buffers with separate real/imag float buffers.

    ONNX does not support complex-valued tensors, so the complex RoPE
    (rotary positional embedding) buffer must be split into its real
    (cosine) and imaginary (sine) components before export.
    """
    # Current SAM3 already registers real/imaginary buffers. Keep freqs_cis:
    # upstream _apply_rope still asserts that the complex buffer exists.
    freqs_cis = getattr(module, "freqs_cis", None)
    if freqs_cis is not None:
        if not hasattr(module, "freqs_cis_real"):
            module.register_buffer("freqs_cis_real", freqs_cis.real.float())
            module.register_buffer("freqs_cis_imag", freqs_cis.imag.float())
        if hasattr(module, "use_rope_real"):
            module.use_rope_real = True
    for child in module.children():
        prepare_rope_buffers_for_onnx(child)


def prepare_fused_mlps_for_onnx(module: torch.nn.Module) -> None:
    """Replace SAM3's inference-only BF16 fused MLP with exportable FP32 ops."""

    def forward(mlp, value):
        value = mlp.fc1(value)
        value = mlp.act(value)
        value = mlp.drop1(value)
        value = mlp.norm(value)
        value = mlp.fc2(value)
        return mlp.drop2(value)

    for child in module.modules():
        if (
            child.__class__.__name__ == "Mlp"
            and child.__class__.__module__ == "sam3.model.vitdet"
        ):
            child.forward = types.MethodType(forward, child)


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
        # Register the backbone as a real child module. Keeping it reachable
        # only through the non-Module processor makes the exporter treat every
        # trainable weight as an invalid requires-grad constant.
        self._backbone = processor.model.backbone
        # Normalise uint8 [0,255] to float [-1, 1] – identical to the
        # reference sam3-onnx export (export_onnx.py).
        self._transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),  # uint8 → float [0,1]
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1,1]
            ]
        )

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # image: (3, H, W) uint8 → normalise → (1, 3, H, W) float
        image = self._transform(image).unsqueeze(0)
        backbone_out = self._backbone._forward_image_no_act_ckpt(image)
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
        self._model: Sam3Image = processor.model

    @torch.no_grad()
    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self._model

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
        # PCS ignores point prompts, but its geometry encoder still needs a
        # well-shaped point sequence when boxes are padded to a fixed capacity.
        self.register_buffer("_point_embedding", torch.zeros(1, 1, 2))
        self.register_buffer("_point_mask", torch.ones(1, 1, dtype=torch.bool))
        self.register_buffer("_point_label", torch.ones(1, 1, dtype=torch.long))

    @torch.no_grad()
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
        geometric_prompt.point_embeddings = self._point_embedding
        geometric_prompt.point_labels = self._point_label
        geometric_prompt.point_mask = self._point_mask
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
        # Export raw heads. Keeping thresholding and original-resolution mask
        # resizing outside ONNX allows runtime confidence and image dimensions
        # to remain fully dynamic, matching the official Sam3Processor logic.
        result = self._model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self._processor.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )
        return (
            result["pred_boxes"],
            result["pred_logits"],
            result["pred_masks"],
            result["presence_logit_dec"],
        )


def export_sam3(
    output_dir: str,
    opset: int = 18,
    simplify_model: bool = False,
    checkpoint_path: str | None = None,
    device: str = "auto",
    max_geometric_prompts: int = 8,
):
    if max_geometric_prompts < 1:
        raise ValueError("max_geometric_prompts must be at least 1")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is None,
        device=device,
    )
    # Replace complex RoPE buffers with float cos/sin – required for ONNX.
    prepare_rope_buffers_for_onnx(model)
    # The current upstream ViT uses a CUDA BF16-only fused addmm/GELU helper.
    # Standard FP32 layers preserve the same MLP semantics and are portable to
    # ONNX Runtime providers.
    prepare_fused_mlps_for_onnx(model)
    processor = Sam3Processor(model)
    model.eval().to(device)

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
    decoder = SAM3Decoder(model, processor).eval().to(device)
    decoder_path = output_dir / "sam3_decoder.onnx"

    # Geometry is sequence-first: [num_marks, batch, coordinates].
    box_coords = torch.zeros(max_geometric_prompts, 1, 4).to(device)
    box_labels = torch.ones(max_geometric_prompts, 1, dtype=torch.long).to(device)
    # Trace one real neutral slot plus right-padding. Tracing with every slot
    # masked can make upstream attention hit an all-masked softmax/FPE.
    box_coords[0, 0] = torch.tensor([0.5, 0.5, 0.01, 0.01], device=device)
    box_masks = torch.ones(1, max_geometric_prompts, dtype=torch.bool).to(device)
    box_masks[0, 0] = False
    orig_h = torch.tensor(1008).to(device)
    orig_w = torch.tensor(1008).to(device)

    # The legacy tracer crashes in torchvision's multi-box RoIAlign symbolic.
    # Dynamo handles the fixed padded geometry path and produces portable ONNX.
    original_pin_memory = torch.Tensor.pin_memory
    original_is_dynamo_compiling = torch.compiler.is_dynamo_compiling
    torch.Tensor.pin_memory = lambda tensor: tensor
    torch.compiler.is_dynamo_compiling = lambda: True
    try:
        torch.onnx.export(
            decoder,
            args=(
                orig_h,
                orig_w,
                vpe0,
                vpe1,
                vpe2,
                fpn0,
                fpn1,
                fpn2,
                l_mask,
                l_feat,
                l_embed,
                box_coords,
                box_labels,
                box_masks,
            ),
            f=str(decoder_path),
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
            output_names=[
                "pred_boxes",
                "pred_logits",
                "pred_masks",
                "presence_logit_dec",
            ],
            opset_version=opset,
            dynamo=True,
        )
    finally:
        torch.Tensor.pin_memory = original_pin_memory
        torch.compiler.is_dynamo_compiling = original_is_dynamo_compiling
    print(f"Saved Decoder to {decoder_path}")

    # ── Simplify models conditionally ─────────────────────────────────────────
    if simplify_model:
        from onnxsim import simplify

        for path in [encoder_path, language_path, decoder_path]:
            print(f"Simplifying {path}...")
            try:
                onnx_model = onnx.load(str(path))
                model_simp, check = simplify(onnx_model)
                assert check, "Simplified ONNX model could not be validated"
                onnx.save(model_simp, str(path))
                print("  → simplified OK")
            except Exception as e:
                print(f"  → simplification failed ({e}), keeping original")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SAM3 to ONNX")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_models/sam3",
        help="Output directory for ONNX models",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX models")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local official SAM3 checkpoint; otherwise download from Hugging Face",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="PyTorch device used during export",
    )
    parser.add_argument(
        "--max-geometric-prompts",
        type=int,
        default=8,
        help="Fixed padded capacity for rectangle/point prompts (default: 8)",
    )
    args = parser.parse_args()
    export_sam3(
        args.output_dir,
        args.opset,
        args.simplify,
        args.checkpoint,
        args.device,
        args.max_geometric_prompts,
    )
