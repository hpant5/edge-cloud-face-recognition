import os
import json
import boto3
import base64
import tempfile
import logging
import traceback
import numpy as np 
from PIL import Image
from facenet_pytorch import MTCNN

log = logging.getLogger()
log.setLevel(logging.INFO)

sqs_client = boto3.client('sqs')

# No default. Failing loudly on a missing env var beats silently writing to the wrong queue.
REQUEST_QUEUE_URL = os.environ['REQUEST_QUEUE_URL']

# Built once per container, not per invocation, so warm starts reuse the loaded model.
_detector = None


def handler(evt, ctx):
    return lambda_handler(evt, ctx)

def lambda_handler(evt, ctx):
    log.info(f"Lambda initiated with request ID: {ctx.aws_request_id}")
    
    try:
        # Parse and validate input first
        req_id, img_content, file_name = extract_request_data(evt)
        log.info(f"Processing req_id: {req_id}, file_name: {file_name}")
        
        # Setup workspace
        workspace = setup_temp_workspace()
        log.info(f"Workspace ready: input={workspace['input']}, output={workspace['output']}")
        
        # Process the image file
        img_input_path = save_image_to_disk(img_content, file_name, workspace['input'])
        log.info(f"Image saved to: {img_input_path}")
        
        # Detect face in image
        detected_face_path = perform_face_detection(img_input_path, workspace['output'])
        
        # Handle no face detected case early
        if detected_face_path is None:
            cleanup_workspace(workspace, img_input_path, None, file_name)
            return build_response(200, {'message': 'No face detected in the image'})
        
        log.info(f"Face detected at: {detected_face_path}")
        
        # Encode and send to queue
        send_face_to_queue(detected_face_path, req_id)
        
        # Cleanup
        cleanup_workspace(workspace, img_input_path, detected_face_path, file_name)
        
        log.info(f"Request {req_id} completed successfully")
        return build_response(200, {'message': 'Face detection successful', 'request_id': req_id})
        
    except Exception as err:
        log.error(f"Error occurred: {str(err)}")
        log.error(f"Traceback: {traceback.format_exc()}")
        return build_response(500, {'error': f'Error processing image: {str(err)}'})


def extract_request_data(evt):
    """Extract and validate request parameters from event"""
    log.info(f"Received event: {json.dumps(evt)}")
    payload = json.loads(evt['body'])
    
    img_content = payload['content']
    req_id = payload['request_id']
    file_name = payload['filename']
    
    return req_id, img_content, file_name


def setup_temp_workspace():
    """Create temporary directories for processing"""
    input_temp_dir = tempfile.mkdtemp()
    output_temp_dir = tempfile.mkdtemp()
    
    return {
        'input': input_temp_dir,
        'output': output_temp_dir
    }


def save_image_to_disk(img_content, file_name, input_dir):
    """Decode base64 image and save to temporary location"""
    # Handle potential directory structure in filename
    dir_part, name_part = os.path.split(file_name)
    
    if dir_part:
        complete_dir_path = os.path.join(input_dir, dir_part)
        os.makedirs(complete_dir_path, exist_ok=True)
        img_path = os.path.join(complete_dir_path, name_part)
    else:
        img_path = os.path.join(input_dir, file_name)
    
    # Decode and write image
    decoded_img = base64.b64decode(img_content)
    with open(img_path, 'wb') as file_handle:
        file_handle.write(decoded_img)
    
    # Verify file exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found after saving at {img_path}")
    
    log.info(f"Image saved, total bytes: {len(decoded_img)}")
    return img_path


def perform_face_detection(img_path, output_dir):
    """Run face detection on input image"""
    log.info("Starting face detection process")

    global _detector
    if _detector is None:
        log.info("Cold start: loading MTCNN")
        _detector = face_detection()

    detected_face_path = _detector.face_detection_func(img_path, output_dir)
    
    if detected_face_path is None:
        log.info("No face detected in image")
    else:
        log.info("Face detection complete")
    
    return detected_face_path


def send_face_to_queue(face_path, req_id):
    """Encode detected face and send to SQS queue"""
    # Read face image
    with open(face_path, 'rb') as file_handle:
        detected_face_data = file_handle.read()
    
    # Encode to base64
    face_encoded = base64.b64encode(detected_face_data).decode('utf-8')
    log.info(f"Face image encoded, length: {len(face_encoded)} chars")
    
    # Build SQS message
    sqs_msg = {
        'request_id': req_id,
        'face': face_encoded,
        'filename': os.path.basename(face_path)
    }
    
    # Send to queue
    log.info(f"Pushing message to queue: {REQUEST_QUEUE_URL}")
    sqs_response = sqs_client.send_message(
        QueueUrl=REQUEST_QUEUE_URL,
        MessageBody=json.dumps(sqs_msg)
    )
    log.info(f"Message pushed successfully, MessageId: {sqs_response.get('MessageId')}")


def cleanup_workspace(workspace, input_file, output_file, file_name):
    """Remove temporary files and directories"""
    log.info("Starting cleanup process")
    
    try:
        # Remove files first
        if input_file and os.path.exists(input_file):
            os.remove(input_file)
            log.info(f"Deleted input file: {input_file}")
        
        if output_file and os.path.exists(output_file):
            os.remove(output_file)
            log.info(f"Deleted face image: {output_file}")
        
        # Handle subdirectory cleanup
        dir_part, _ = os.path.split(file_name)
        if dir_part:
            for root_dir, subdirs, files in os.walk(workspace['input'], topdown=False):
                for subdir_name in subdirs:
                    subdir_path = os.path.join(root_dir, subdir_name)
                    os.rmdir(subdir_path)
                    log.info(f"Deleted directory: {subdir_path}")
        
        # Remove main directories
        os.rmdir(workspace['input'])
        log.info(f"Deleted input directory: {workspace['input']}")
        
        os.rmdir(workspace['output'])
        log.info(f"Deleted output directory: {workspace['output']}")
        
    except Exception as cleanup_err:
        log.warning(f"Cleanup error: {str(cleanup_err)}")


def build_response(status_code, body_content):
    """Build standardized Lambda response"""
    return {
        'statusCode': status_code,
        'body': json.dumps(body_content)
    }

    
class face_detection:
    # Face detection using InceptionResnetV1 with pretrained weights
    def __init__(self):
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    def face_detection_func(self, img_path, dest_path):
        # Load and prepare image
        image = self._load_and_prepare_image(img_path)
        identifier = self._extract_identifier(img_path)
        
        # Run detection
        detected, probability = self.mtcnn(image, return_prob=True, save_path=None)
        print(f"MTCNN executed, detection probability: {probability}")
        
        # Process result
        if detected is None:
            print("No face is detected")
            return None
        
        return self._save_detected_face(detected, dest_path, identifier)
    
    def _load_and_prepare_image(self, img_path):
        """Load image and convert to proper format"""
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        image = Image.fromarray(image)
        print("Image loaded and converted to proper format, ready for MTCNN")
        return image
    
    def _extract_identifier(self, img_path):
        """Extract base identifier from image path"""
        return os.path.splitext(os.path.basename(img_path))[0].split(".")[0]
    
    def _save_detected_face(self, detected, dest_path, identifier):
        """Normalize and save detected face tensor"""
        os.makedirs(dest_path, exist_ok=True)
        
        # Normalize face tensor
        normalized_face = detected - detected.min()
        normalized_face = normalized_face / normalized_face.max()
        normalized_face = (normalized_face * 255).byte().permute(1, 2, 0).numpy()
        print("Face tensor normalized and converted, preprocessing complete")
        
        # Save as image
        face_image = Image.fromarray(normalized_face, mode="RGB")
        output_file_path = os.path.join(dest_path, f"{identifier}_face.jpg")
        face_image.save(output_file_path)
        print(f"Face saved successfully at {output_file_path}")
        
        return output_file_path
