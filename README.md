# Edge-to-Cloud Face Recognition Pipeline

A two-stage inference pipeline. Face **detection** runs on an edge device, face **recognition** runs in the cloud on containerised PyTorch, and the two stages are joined by asynchronous SQS queues rather than by a direct call.

The point of the split is that detection is cheap and recognition is expensive. Doing detection at the edge means the only thing that ever crosses the network is a 240x240 cropped face instead of a full frame, and frames with no face in them never reach the cloud at all.

## Architecture

```mermaid
flowchart LR
    A[IoT device] -->|MQTT: base64 frame| B[Greengrass component<br/>MTCNN detection]
    B -->|no face| E[(response queue)]
    B -->|cropped face| C[(request queue)]
    C -->|SQS trigger| D[Lambda on ECR<br/>InceptionResnetV1 / vggface2]
    D -->|match name| E
    E --> F[Client polls result]
```

The device publishes a base64 frame to an MQTT topic. A Greengrass component subscribed to that topic decodes it in memory, runs MTCNN, and takes one of two paths:

- **No face found.** It writes a `No-Face` result straight to the response queue and stops. The cloud is never invoked.
- **Face found.** It normalises the face tensor, encodes it as a JPEG, and puts it on the request queue.

The request queue triggers a Lambda running a container image out of ECR. That function holds `InceptionResnetV1` with `vggface2` weights in module scope, embeds the incoming face, does a nearest-neighbour lookup against a precomputed embedding store, and writes the matched name to the response queue.

## Design decisions

**Why queues between the stages rather than a direct invoke.** If detection called recognition synchronously, detection would be held open for the whole recognition latency, including any cold start on the PyTorch container. Ingestion throughput would then be capped by the slowest stage. With a queue in between, detection returns as soon as the message is enqueued, and recognition drains the backlog at whatever rate its concurrency allows. A slow or scaling-up recognition stage produces queue depth, not dropped or blocked ingestion.

**Why a separate response queue rather than a callback.** The client never has to hold a connection open, and results from both the edge short-circuit path and the cloud path land in the same place with the same shape. The consumer does not need to know which path produced a given result.

**Why detection runs at the edge.** A cropped 240x240 face is a small fraction of the bytes of the source frame, so the upload is smaller. More importantly, frames with no face in them are rejected before they cost a Lambda invocation. On any realistic camera feed that is most frames.

**Model loading.** The recognition model and the embedding store are built once at module scope, not per invocation, so a warm container reuses them. The same applies to the MTCNN instance in the Greengrass component.

## Repository layout

```
face-detection/
  fd_component.py   Greengrass component. MQTT subscriber, in-memory MTCNN, edge short-circuit.
  fd_lambda.py      Cloud-only variant. Same detection behind an HTTP-invoked Lambda,
                    kept for comparison against the edge path.
face-recognition/
  fr_lambda.py      SQS-triggered Lambda. Embedding and nearest-neighbour match.
```

`fd_component.py` and `fd_lambda.py` are two implementations of the same detection step, one at the edge and one in the cloud. The edge one is the one the architecture diagram describes.

## Running it

Both Lambdas ship as container images because `facenet-pytorch` and its torch dependency are far past the Lambda zip layer limit.

```bash
# build and push the recognition image
docker build -t face-recognition ./face-recognition
aws ecr get-login-password --region <region> \
  | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag face-recognition:latest <account>.dkr.ecr.<region>.amazonaws.com/face-recognition:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/face-recognition:latest
```

Configuration is by environment variable:

| Variable | Used by | Purpose |
|---|---|---|
| `REQUEST_QUEUE_URL` | detection | Queue the cropped face is written to |
| `RESPONSE_QUEUE_URL` | recognition, edge short-circuit | Queue results are written to |
| `MQTT_TOPIC` | Greengrass component | Topic the device publishes frames on |

The recognition function expects an embedding store at `face-recognition/data.pt`, a two-element tensor file of `[embeddings, names]`. It is not in this repo. Generate it by running `InceptionResnetV1(pretrained='vggface2')` over your own labelled face images and saving the pair with `torch.save`.

## Known limitations

I would rather list these than have you find them.

- **SQS message size.** The cropped face travels inside the message body as base64, which inflates it by about a third. A 240x240 JPEG stays well inside the 256 KB SQS limit in practice, but there is no explicit guard, and the right fix at larger crop sizes is to put the image in S3 and send the key.
- **At-least-once delivery.** SQS can redeliver, and the recognition handler is not idempotent, so a single face can be recognised twice and write two results. For this workload that is harmless because the result is deterministic and keyed by `request_id`, but a consumer that counts results would need to deduplicate.
- **No dead-letter queue.** A message that fails repeatedly currently retries until the retention period expires. A DLQ on the request queue is the obvious next step.
- **No infrastructure as code.** The queues, functions, triggers and Greengrass deployment were created by hand. This should be Terraform or SAM.
- **No throughput numbers.** I have not benchmarked this under sustained load, so I am not going to quote figures for it. The design intent above is what the architecture supports, not a measured result.
