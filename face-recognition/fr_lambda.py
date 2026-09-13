import os
import json
import boto3
import base64
import io
import logging
import traceback
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1

log = logging.getLogger()
log.setLevel(logging.INFO)

os.environ['TORCH_HOME'] = '/tmp'

sqs_client = boto3.client('sqs', region_name=os.environ.get('AWS_REGION', 'us-east-1'))

# No default. Failing loudly on a missing env var beats silently writing to the wrong queue.
RESPONSE_QUEUE_URL = os.environ['RESPONSE_QUEUE_URL']

recognition_model = InceptionResnetV1(pretrained='vggface2').eval()

def load_embeddings_database():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'data.pt')
        loaded_data = torch.load(model_path, map_location=torch.device('cpu'))
        embeddings = loaded_data[0]
        names = loaded_data[1]
        log.info("Successfully loaded embeddings database")
        return embeddings, names
    except Exception as err:
        log.error(f"Failed to load embeddings: {str(err)}")
        return [], []

stored_embeddings, stored_names = load_embeddings_database()

def lambda_handler(event, context):
    log.info(f"Lambda started with request ID: {context.aws_request_id}")
    try:
        results = process_sqs_batch(event.get('Records', []))
        log.info(f"Face recognition batch completed. Processed {len(results)} records")
        return {'statusCode': 200, 'body': json.dumps({'message': 'Success'})}
    except Exception as err:
        log.error(f"Error: {str(err)}")
        return {'statusCode': 500, 'body': json.dumps({'error': str(err)})}

def process_sqs_batch(records):
    results = []
    for sqs_record in records:
        try:
            result = process_single_record(sqs_record)
            results.append(result)
        except Exception as err:
            log.error(f"Error processing record: {str(err)}")
    return results

def process_single_record(sqs_record):
    msg_payload = json.loads(sqs_record['body'])
    req_id = msg_payload.get('request_id')
    face_b64 = msg_payload['face']
    file_name = msg_payload.get('filename', 'unknown.jpg')
    identified_name = recognize_face_from_base64(face_b64)
    send_result_to_queue(req_id, identified_name)
    return {'request_id': req_id, 'result': identified_name}

def recognize_face_from_base64(face_b64):
    try:
        face_bytes = base64.b64decode(face_b64)
        img_buffer = io.BytesIO(face_bytes)
        pil_image = Image.open(img_buffer).convert("RGB").resize((160, 160))
        np_array = np.array(pil_image, dtype=np.float32)
        np_array = (np_array - 127.5) / 128.0
        np_array = np.transpose(np_array, (2, 0, 1))
        tensor_data = torch.tensor(np_array, dtype=torch.float32)
        face_embedding = recognition_model(tensor_data.unsqueeze(0)).detach()
        if not stored_embeddings:
            return "unknown"
        distances = [torch.dist(face_embedding, emb).item() for emb in stored_embeddings]
        min_idx = distances.index(min(distances))
        return stored_names[min_idx]
    except Exception as err:
        log.error(f"Recognition failed: {str(err)}")
        return "error"

def send_result_to_queue(req_id, identified_name):
    result_msg = {'request_id': req_id, 'result': identified_name.strip() if identified_name else "unknown"}
    sqs_client.send_message(QueueUrl=RESPONSE_QUEUE_URL, MessageBody=json.dumps(result_msg))
    log.info(f"Completed req_id {req_id}. Identified: {identified_name}")
