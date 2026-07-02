# Models

## Edge model (on-robot)

`export_onnx.py` exports torchvision's `ssdlite320_mobilenet_v3_large`
(COCO-pretrained) to ONNX. It is small and fast, meant to run on the robot's
own CPU with no network.

```bash
pip install torch torchvision onnx onnxruntime
python export_onnx.py --out edge_ssdlite.onnx
```

This writes `edge_ssdlite.onnx` and `edge_ssdlite_labels.txt` (COCO categories).

### Export caveats

- Input is fixed at 320x320 so edge latency is predictable for a real-time loop.
  The detector class letterboxes by simple resize and rescales boxes back to the
  original resolution.
- torchvision detection models emit a variable number of detections, so the
  ONNX graph uses a dynamic `num_detections` axis on the outputs.
- If you target a Jetson, add the CUDA or TensorRT execution provider in
  `edge_detector.py` and re-benchmark; CPU numbers are the honest baseline.

## Cloud model (SageMaker)

The cloud endpoint runs `fasterrcnn_resnet50_fpn`, a heavier and more accurate
detector, in `cloud/sagemaker/inference.py`. Same COCO label space as the edge
model so their outputs are directly comparable. That comparability is what the
router and the benchmark rely on.
