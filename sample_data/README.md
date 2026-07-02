# Sample data

Drop a handful of `.jpg` or `.png` images here to run the benchmark:

```bash
cd ../scripts
python benchmark.py --images ../sample_data \
  --model ../models/edge_ssdlite.onnx \
  --labels ../models/edge_ssdlite_labels.txt
```

Any everyday photos work (people, vehicles, furniture). A good demo mixes easy
images (a clear single object, which the edge model handles alone) with hard
ones (small, cluttered, or occluded objects, which trigger escalation). That
contrast is what shows the router doing its job.

Images are not committed to keep the repo light.
