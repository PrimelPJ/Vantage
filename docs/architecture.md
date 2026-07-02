# Vantage architecture notes

## The routing policy

`router_policy.should_escalate(edge_top_score, edge_num_dets, latency_budget)`
returns `(escalate?, reason)`. The gates run in order:

1. **Latency gate.** If this frame's latency budget is tighter than the
   configured minimum, stay on the edge. A tight control loop must not block on
   a network round trip.
2. **Confidence gate.**
   - `edge_top_score >= conf_high` (default 0.70): trust the edge, do not escalate.
   - score in `[conf_low, conf_high)` or a non-empty scene below `conf_low`:
     doubtful, and a candidate for escalation.
   - empty scene at a very low score: nothing to escalate for.
3. **Budget gate.** A token bucket (`cloud_calls_per_sec`, `cloud_burst`) caps
   escalations. If the bucket is empty, stay on the edge this frame. The reason
   is recorded as `cloud_budget_exhausted` so you can see backpressure in the
   metrics.

Every decision records a reason, which is what turns the metrics into a story:
you can see exactly why frames stayed local or went to the cloud.

## Shared detection format

```json
{ "label": "person", "score": 0.83, "box": [x1, y1, x2, y2] }
```

Edge and cloud both emit a list of these in COCO label space. Because the
formats match, the router treats them as interchangeable and the benchmark can
compute agreement between them.

## Data flywheel

```
uncertain frame ─▶ S3 (hard-cases/dt=YYYY-MM-DD/<device>/<ts>.jpg + .json)
                       │
                       ▼
                 label + curate
                       │
                       ▼
             fine-tune edge model
                       │
                       ▼
        escalation rate falls ─▶ cloud cost falls
```

The captured JSON stores both the edge detections and, when available, the
cloud detections for the same frame. Cloud-vs-edge disagreement is a strong,
cheap label signal for what to fix.

## Security: how a robot calls SageMaker without static keys

The robot authenticates to AWS IoT Core with its X.509 certificate, then uses
the IoT Core credentials provider to exchange that cert for temporary IAM role
credentials (the `vantage-robot` role in Terraform). Those short-lived creds
allow exactly two actions: `sagemaker:InvokeEndpoint` on `vantage-*` endpoints
and `s3:PutObject` to the hard-case bucket. No long-lived AWS keys ever live on
the robot.

## Latency and cost intuition

- Edge inference: single-digit to low-tens of milliseconds on CPU, zero
  marginal cost (owned compute).
- Cloud inference: model time plus a network round trip, and a per-invocation
  dollar cost.
- Hybrid latency tracks edge latency for the majority of frames and only pays
  the cloud tail on the escalated fraction. If escalation rate is 15 percent,
  you pay roughly 15 percent of the all-cloud cost and stay near edge latency on
  the other 85 percent.
