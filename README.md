# Vantage

An edge-versus-cloud object detection router for ROS2 robots. A small detector
runs on the robot for every frame; a per-frame policy decides when the result
is uncertain and escalates just those frames to a heavier model on an AWS
SageMaker endpoint. Uncertain frames are captured to S3 to build a retraining
dataset, closing the loop.

> **The one-sentence thesis.** Running a big model in the cloud on every frame
> is accurate but slow and expensive; running a small model on the robot is fast
> and free but misses hard cases. Vantage spends cloud compute only on the
> frames that need it, and turns those same frames into training data so the
> edge model keeps getting better.

Runs fully on a laptop with no ROS and no AWS through the benchmark script, so
you can show real numbers in an interview.

---

## Architecture

```
   ROS2 robot                                              AWS
  ┌────────────────────────────┐
  │  /camera/image_raw          │
  │            │                │
  │            ▼                │
  │   EdgeDetector (ONNX)       │   every frame, on-robot CPU, no network
  │   SSDLite MobileNetV3       │
  │            │ dets + score   │
  │            ▼                │
  │   Router policy             │
  │   confident? ── yes ──▶ publish edge result
  │       │ no (uncertain)      │
  │       ▼                     │   only uncertain frames
  │   escalate (rate-limited) ──┼───────────────▶ ┌────────────────────┐
  │            ▲                │  JPEG over HTTPS │ SageMaker endpoint  │
  │            │ cloud dets     │◀────────────────│ FasterRCNN ResNet50 │
  │            ▼                │                  └────────────────────┘
  │   publish best result       │
  │   (vision_msgs/Detection2DArray)
  │            │                │
  │            ▼                │   uncertain frames -> retraining dataset
  │   HardCaseCapture ──────────┼───────────────▶ ┌────────────────────┐
  │   (background thread)       │                  │ S3 (date/conf keyed)│
  └────────────────────────────┘                  └─────────┬──────────┘
                                                            │
                                    fine-tune edge model on its own hard cases
                                    -> escalation rate and cloud cost fall
```

See `docs/architecture.md` for the routing policy in detail and the data
flywheel.

---

## Repository layout

```
vantage/
  models/
    export_onnx.py               export the edge model (SSDLite) to ONNX
  ros2_ws/src/vantage_perception/
    vantage_perception/
      detection.py               shared Detection format (edge == cloud)
      edge_detector.py           ONNX Runtime on-robot inference
      cloud_client.py            SageMaker invoke + a local cloud stand-in
      router_policy.py           the escalation decision (the core of the repo)
      hard_case_capture.py       async S3 upload of uncertain frames
      metrics.py                 latency, escalation rate, cost accounting
      inference_router.py        the ROS2 node wiring it together
    launch/perception.launch.py
  cloud/
    sagemaker/
      inference.py               SageMaker inference contract (heavy model)
      deploy_endpoint.py         package + deploy the endpoint
    infra/                       Terraform: S3, IAM roles
  scripts/
    benchmark.py                 edge vs cloud vs hybrid, no ROS/AWS needed
  docker/                        ROS2 Humble container for the router
  sample_data/                   drop test images here
```

---

## Quickstart

### Path A: benchmark on a laptop (no ROS, no AWS)

This is the fastest way to see the whole idea working.

```bash
pip install torch torchvision onnx onnxruntime pillow numpy

# 1. Export the edge model
cd models
python export_onnx.py --out edge_ssdlite.onnx

# 2. Add a few images to sample_data/, then benchmark
cd ../scripts
python benchmark.py --images ../sample_data \
  --model ../models/edge_ssdlite.onnx \
  --labels ../models/edge_ssdlite_labels.txt
```

You get a table of edge vs cloud vs hybrid latency, the escalation rate, an
estimated cost comparison, and a top-label agreement score, plus a CSV. The
cloud path uses a local heavier model with synthetic network latency, so the
comparison is real without deploying anything.

### Path B: real ROS2 robot, edge-only

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# any camera publishing sensor_msgs/Image on /camera/image_raw works
ros2 run vantage_perception router --ros-args \
  -p model_path:=$(pwd)/../models/edge_ssdlite.onnx \
  -p labels_path:=$(pwd)/../models/edge_ssdlite_labels.txt

# view detections
ros2 topic echo /vantage/detections
```

### Path C: add the cloud

```bash
# 1. Provision S3 + IAM
cd cloud/infra
terraform init && terraform apply

# 2. Deploy the heavy model to a SageMaker endpoint
cd ../sagemaker
python deploy_endpoint.py \
  --role $(terraform -chdir=../infra output -raw sagemaker_role_arn) \
  --bucket <a-model-artifact-bucket> \
  --endpoint vantage-detector

# 3. Run the router with escalation + capture enabled
ros2 run vantage_perception router --ros-args \
  -p model_path:=.../edge_ssdlite.onnx \
  -p labels_path:=.../edge_ssdlite_labels.txt \
  -p sagemaker_endpoint:=vantage-detector \
  -p hard_case_bucket:=$(terraform -chdir=../infra output -raw hard_case_bucket)
```

---

## Design decisions

**Edge first, escalate on doubt.** Every frame runs on the edge model. The
router escalates only when the edge is uncertain: the top confidence sits in an
ambiguous band, or a non-empty scene comes back weak. Confident frames never
pay for the cloud. This is the whole cost argument and it lives in one function,
`router_policy.should_escalate`.

**Escalation is rate-limited.** A token bucket caps cloud calls per second, so a
persistently hard scene degrades to edge-only rather than running up an
unbounded SageMaker bill. Backpressure by design, not by surprise.

**A hard latency gate.** If a control loop cannot spare a round trip this frame,
the router stays on the edge regardless of confidence. Safety-critical
perception never blocks on the network.

**Edge and cloud speak the same format.** Both return the same `Detection`
list in COCO label space. That symmetry is what lets the router swap between
them transparently and lets the benchmark measure agreement between the two.

**Uncertain frames are the dataset.** The frames the edge model struggled with
are exactly the ones worth labeling. Capturing them to S3, keyed by date and
device, builds a targeted retraining set. Fine-tune the edge model on its own
hard cases and the escalation rate (and cost) drops over time. That is the
flywheel.

**Honest baselines.** The edge detector runs on CPU by default, which is what a
robot without a GPU actually has. The cloud stand-in adds synthetic network
latency so laptop benchmarks are not misleadingly fast.

---

## Talking points for interviews

- Why not run the big model in the cloud for everything? (latency, cost, and a
  network dependency in the perception loop; quantify it with the benchmark)
- Why not run everything on the edge? (accuracy on hard frames; show the
  agreement gap between edge-only and cloud)
- How does the router decide? (confidence band, latency gate, token-bucket
  budget, walk through `should_escalate`)
- How do you stop cloud costs from exploding on a hard scene? (rate limit,
  graceful degradation to edge)
- How does the system improve over time? (hard-case capture to S3, fine-tune
  on captured data, escalation rate falls)
- How does a robot call SageMaker without static credentials? (IoT Core
  credentials provider: X.509 cert exchanged for temporary IAM role creds)
- Where would you put this on real hardware? (ONNX Runtime with a TensorRT
  provider on a Jetson; the code already isolates the provider choice)

---

## Extending it

- Fine-tune the edge model on captured hard cases and track escalation rate
  falling release over release.
- Add a SageMaker Async endpoint for batch re-scoring of a whole capture day.
- Replace the top-label agreement proxy with mAP against a labeled holdout for
  a rigorous accuracy number.
- Run the edge model under Greengrass V2 so model updates deploy over the air.

## License

MIT. See `LICENSE`.
