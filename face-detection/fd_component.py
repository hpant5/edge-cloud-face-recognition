import os
import time
import awsiot.greengrasscoreipc
import awsiot.greengrasscoreipc.client as gg_client

from awsiot.greengrasscoreipc.model import SubscribeToTopicRequest, SubscriptionResponseMessage

import json
import base64
import logging
from io import BytesIO

import boto3
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Configuration comes from the Greengrass component recipe.
# No defaults: failing loudly on a missing value beats writing to the wrong queue.
REQUEST_QUEUE_URL = os.environ["REQUEST_QUEUE_URL"]
RESPONSE_QUEUE_URL = os.environ["RESPONSE_QUEUE_URL"]
TOPIC = os.environ["MQTT_TOPIC"]

sqs_client = boto3.client("sqs", region_name=os.environ.get("AWS_REGION", "us-east-1"))

# Built once at module scope so every message reuses the loaded model.
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

TIMEOUT = 10


def decode_image(encoded_str: str) -> Image.Image:
    """Decode base64 string into a RGB PIL image (no disk)."""
    raw = base64.b64decode(encoded_str)
    return Image.open(BytesIO(raw)).convert("RGB")


def detect_face(image: Image.Image):
    """Run MTCNN in memory and return face tensor or None."""
    face_tensor, prob = mtcnn(image, return_prob=True, save_path=None)
    LOGGER.info("MTCNN probability: %s", prob)
    if face_tensor is None:
        LOGGER.info("No face detected")
        return None
    return face_tensor


def face_tensor_to_base64(face_tensor, identifier: str) -> tuple:
    """Convert face tensor to JPEG in memory and return (b64, filename)."""
    arr = face_tensor - face_tensor.min()
    arr = arr / arr.max()
    arr = (arr * 255).byte().permute(1, 2, 0).numpy()

    img = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    encoded_face = base64.b64encode(buf.read()).decode("utf-8")
    filename = f"{identifier}_face.jpg"
    return encoded_face, filename


def send_face_to_request_queue(request_id: str, encoded_face: str, filename: str):
    """Send detected face payload to the request SQS queue for Lambda processing."""
    msg = {
        "request_id": request_id,
        "face": encoded_face,
        "filename": filename,
    }
    LOGGER.info("Sending face to request queue: %s", REQUEST_QUEUE_URL)
    sqs_client.send_message(
        QueueUrl=REQUEST_QUEUE_URL,
        MessageBody=json.dumps(msg),
    )
    LOGGER.info("Face sent to request queue for request_id=%s", request_id)


def send_no_face_to_response_queue(request_id: str, filename: str):
    """BONUS: Send No-Face result directly to response queue (skip Lambda)."""
    msg = {
        "request_id": request_id,
        "filename": filename,
        "result": "No-Face",
    }
    LOGGER.info("No face detected - sending directly to response queue: %s", RESPONSE_QUEUE_URL)
    sqs_client.send_message(
        QueueUrl=RESPONSE_QUEUE_URL,
        MessageBody=json.dumps(msg),
    )
    LOGGER.info("No-Face result sent to response queue for request_id=%s", request_id)


def process_request(encoded: str, request_id: str, filename: str):
    """End-to-end in-memory pipeline for one frame."""
    image = decode_image(encoded)
    face_tensor = detect_face(image)
    
    if face_tensor is None:
        # BONUS: No face detected - send directly to response queue
        send_no_face_to_response_queue(request_id, filename)
        return

    # Face detected - send to request queue for Lambda processing
    base_name = filename.rsplit(".", 1)[0]
    encoded_face, face_out_name = face_tensor_to_base64(face_tensor, base_name)
    send_face_to_request_queue(request_id, encoded_face, face_out_name)


def handle_mqtt_message(topic: str, payload_bytes: bytes):
    """Handle one MQTT message from the client device."""
    try:
        payload_str = payload_bytes.decode("utf-8")
        data = json.loads(payload_str)

        encoded = data["encoded"]
        request_id = data["request_id"]
        filename = data["filename"]

        LOGGER.info(
            "Received MQTT message: topic=%s, request_id=%s, filename=%s",
            topic,
            request_id,
            filename,
        )

        process_request(encoded, request_id, filename)
        LOGGER.info("Finished processing request_id=%s", request_id)
    except Exception as exc:
        LOGGER.exception("Error handling MQTT message: %s", exc)


class StreamHandler(gg_client.SubscribeToTopicStreamHandler):
    def __init__(self):
        super().__init__()

    def on_stream_event(self, event: SubscriptionResponseMessage) -> None:
        try:
            payload_bytes = event.binary_message.message
            topic = event.binary_message.context.topic if hasattr(event.binary_message, 'context') else TOPIC
            handle_mqtt_message(topic, payload_bytes)
        except Exception as exc:
            LOGGER.exception("Error in on_stream_event: %s", exc)

    def on_stream_error(self, error: Exception) -> bool:
        LOGGER.exception("Stream error: %s", error)
        return False  # keep stream open

    def on_stream_closed(self) -> None:
        LOGGER.info("Subscribe-to-topic stream closed")


def main():
    print("FaceDetection component starting...", flush=True)
    LOGGER.info("Starting FaceDetection Greengrass component (IPC subscriber)")
    
    try:
        ipc_client = awsiot.greengrasscoreipc.connect()
        request = SubscribeToTopicRequest()
        request.topic = TOPIC

        handler = StreamHandler()
        operation = ipc_client.new_subscribe_to_topic(handler)
        operation.activate(request)
        future_response = operation.get_response()

        future_response.result(TIMEOUT)
        LOGGER.info("Successfully subscribed to topic: %s", TOPIC)
        print(f"Successfully subscribed to topic: {TOPIC}", flush=True)

        # Keep the component alive to receive messages
        while True:
            time.sleep(10)

    except Exception as exc:
        LOGGER.exception("Fatal error in main: %s", exc)
        raise


if __name__ == "__main__":
    main()
