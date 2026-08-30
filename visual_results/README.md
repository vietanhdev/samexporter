# Visual model results

This directory contains real-model inference evidence for human review. Open the
PNGs at full size to inspect boundaries, prompt placement, holes, and leakage.
Blue is the selected mask for single-result families, green is a positive
point/rectangle, and red is a negative point. SAM3 multi-instance results use
distinct colors, contours, and numbers. Mock sessions are used only by isolated
unit tests; no mocked image is presented here as quality evidence.

[Open the cross-model contact sheet](model_comparison.jpg) for a quick visual
comparison, then inspect the linked full-resolution PNGs below.

## Current verification status

| Family | Real artifacts produced in this change | Validation |
|---|---|---|
| SAM ViT-B | [truck box](sam_vit_b/truck_box.png), [truck point](sam_vit_b/truck_point.png), [plants box](sam_vit_b/plants_box.png), [plants point](sam_vit_b/plants_point.png), [refined plant](sam_vit_b/plants_box_refined.png) | Official checkpoint, pinned source export, CPU ONNX |
| MobileSAM | [truck box](mobile_sam/truck_box.png), [truck point](mobile_sam/truck_point.png), [plants box](mobile_sam/plants_box.png), [plants point](mobile_sam/plants_point.png), [refined plant](mobile_sam/plants_box_refined.png) | Official checkpoint, newly exported encoder and decoder, CPU ONNX |
| EfficientSAM-Ti | [truck box](efficient_sam_ti/truck_box.png), [truck point](efficient_sam_ti/truck_point.png), [plants box](efficient_sam_ti/plants_box.png), [plants point](efficient_sam_ti/plants_point.png), [refined plant](efficient_sam_ti/plants_box_refined.png) | Official Apache-2.0 split ONNX files, CPU ONNX |
| SAM 2.1 Tiny | [truck box](sam2_1_tiny/truck_box.png), [truck point](sam2_1_tiny/truck_point.png), [plants box](sam2_1_tiny/plants_box.png), [plants point](sam2_1_tiny/plants_point.png), [refined plant](sam2_1_tiny/plants_box_refined.png) | Official checkpoint, pinned source export, CPU ONNX |
| SAM 3 | [truck text](sam3/truck_text.png), [truck box selection](sam3/truck_box.png), [truck text + point](sam3/truck_point.png), [all 20 plants](sam3/plants_text.png), [top five plants](sam3/plants_text_top5.png), [plants box selection](sam3/plants_box.png), [two-mark selection](sam3/plants_box_refined.png) | Latest official gated checkpoint, three newly exported ONNX graphs, official-style mask NMS, CPU ONNX |

Older checked-in references remain under `reference_sam/` and
`reference_sam2/`. They are retained for comparison but are not counted as
evidence that the current exporters ran successfully.

## Numerical checks

- SAM ViT-B PyTorch/ONNX mask IoU: truck box 0.9921, truck point 1.0000,
  plants box 0.9858, plants point 1.0000. Encoder embeddings had approximately
  `1e-7` mean absolute error and cosine similarity 1.0000.
- MobileSAM PyTorch/ONNX mask IoU: truck box 0.9961, truck point 0.9188,
  plants box 0.9962, plants point 0.9945. The lower truck-point agreement is an
  ambiguous single-click/TinyViT case; both results visibly select the door.
- SAM 2.1 Tiny PyTorch/ONNX mask IoU: truck box 0.99757, truck point 0.99576,
  plants box 0.99681, plants point 0.99761.
- EfficientSAM-Ti mask area fractions: truck box 0.2921, truck point 0.0366,
  plants box 0.0324, plants point 0.0104. Both positive points are contained;
  box-mask pixels outside the prompt are 0.14% for truck and 0.02% for plants.
- SAM 3 real CPU CLI runs retain one mask for truck text/box/text-plus-point,
  one mask for each plant geometry selection, 20 masks for text-only `plant`,
  and five when capped with `--max_instances 5`. Complete cold-process runs took
  18.9–31.8 seconds and peaked at 8.2–8.5 GiB resident memory on this host.

The refined portrait prompt combines a tight rectangle with a positive point on
the pot. Every tested family selects the intended whole plant/pot. SAM3 geometry
still acts as a visual concept exemplar internally; the CLI's default `auto`
mode now ranks prompt overlap and returns the best local match, while
`--sam3_output_mode all` retains the broader concept-discovery behavior.

## Run logs

- [SAM 3 current export](sam3/export_current.log)
- [SAM 3 text run and resource usage](sam3/truck_text.log)
- [SAM 3 plant box selection](sam3/plants_box.log)
- [SAM 3 refined plant selection](sam3/plants_box_refined.log)
- [SAM 3 all-plant text discovery](sam3/plants_text.log)
- [SAM 3 capped top-five discovery](sam3/plants_text_top5.log)
- [SAM 3 packaged-tokenizer real-model validation](sam3/tokenizer_validation.log)
- [EfficientSAM refined portrait timing](efficient_sam_ti/plants_box_refined.log)
- Earlier SAM3 failure logs remain as debugging evidence for the prompt-layout,
  fixed-token, and missing-temporary-artifact failures found before the final
  exporter and validation runs.

## Reproduce

```bash
bash download_all_models.sh
bash test_comprehensive.sh
```

Every future test writes both a PNG and sibling `.log` under
`visual_results/runs/` by default. Set `SAMEXPORTER_RESULTS_DIR` to another
persistent path when comparing runs. Promote only reviewed representative
outputs into a named family directory.
