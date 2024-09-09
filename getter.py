import sys
import base64
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from werkzeug.exceptions import BadRequest
from fastapi import FastAPI, HTTPException
from typing import Any, Dict, Tuple
from pydantic import BaseModel
from ultralytics import YOLO
from tiller_light import tiller_light

model_object_detection = YOLO('best.pt')

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=1)
def isBase64(sb):
        try:
                if isinstance(sb, str):
                        # If there's any unicode here, an exception will be thrown and the function will return false
                        sb_bytes = bytes(sb, 'ascii')
                elif isinstance(sb, bytes):
                        sb_bytes = sb
                else:
                        raise ValueError("Argument must be string or bytes")
                return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
        except Exception:
                return False



def nms_tf(bboxes, pscores, threshold):
    '''
    https://github.com/satheeshkatipomu/nms-python/blob/master/NMS%20using%20Python%2C%20Tensorflow.ipynb
    NMS: NMS using in-built tf.image.non_max_suppression(bboxes,scores,top_n_proposal_after_nms,iou_threshould)

    Input:
        bboxes(tensor of bounding proposals) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max)
        threshold(float): Overlapping threshold above which proposals will be discarded.

    Output:
        filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold.
    '''
    tf_format_boxes = []#bboxes are format (xmin, ymax, width, height)
    for box in bboxes:#tf wants boxes on format (y_min, x_min, y_max, x_max)
        tf_format_boxes.append(np.array((box[1]+box[3], box[0], box[1], box[0]+box[2])))
    bboxes = tf.convert_to_tensor(tf_format_boxes)
    bbox_indices = tf.image.non_max_suppression(tf_format_boxes, pscores, 100, iou_threshold = threshold)
    filtered_bboxes = tf.gather(bboxes,bbox_indices)#Henter ut kun de bboxes som forekommer i bbox_indices
    scores = tf.gather(pscores,bbox_indices)
    return filtered_bboxes.numpy(), scores.numpy()



def image_handler(image_bytes: bytes, pred_conf_limit: float, tiles: Tuple[int, int], overlap_percentage: float, frame_number: int) -> Dict[str, Any]:
    """
    base64_string = request.json['image']


    if isBase64(base64_string):
        pass
    else:
        print('non base64 string recieved')
        raise BadRequest
    #base64_string.encode('utf-8')
    img_data = base64.b64decode(base64_string)
    """    
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #imdecode returns BGR fomat, yolo needs rgb

    frames = tiller_light(img)
    detections = []
    scores = []
    for i, elem in enumerate(frames):
        frame = elem[0]
        x1,y2 = elem[1]

        res = model_object_detection(frame)
        #boxes_normalized = res[0].boxes.numpy().xywhn.tolist()
        boxes_coords = res[0].boxes.numpy().xywh.tolist()
        confidence = res[0].boxes.conf.tolist()

        for e, conf in zip(boxes_coords, confidence):
            detections.append((int(e[0]-(e[2]/2)+x1), int(e[1]-(e[3]/2)+y2), int(e[2]), int(e[3])))
            scores.append(conf)

    #passing the whole image
    res = model_object_detection(img)
    boxes_coords = res[0].boxes.numpy().xywh.tolist()
    confidence = res[0].boxes.conf.tolist()
    for e, conf in zip(boxes_coords, confidence):
        detections.append((int(e[0]-(e[2]/2)), int(e[1]-(e[3]/2)), int(e[2]), int(e[3])))
        scores.append(conf)

            
    #NMS
    print(len(detections))
    unique_detections, unique_scores = nms_tf(detections, scores, 0.2)


    final_box, final_conf = [], []
    for box, conf in zip(unique_detections, unique_scores):
        if conf > pred_conf_limit:
            final_box.append((int(box[1]), int(box[0]), int(box[3]-box[1]), int(box[0]-box[2])))
            final_conf.append(float(conf))

    print('returning!')
    return {
        'status': 'success',
        'message': "Image processed sucessfully",
        'detections': final_box, #if on gpu, add .cpu() after boxes
        'confidence': final_conf,
        'frame_number': frame_number
        }



class ImageRequest(BaseModel):
    image_base64: str
    pred_conf_limit: float
    overlap_percentage: float
    tiles: Tuple[int, int]
    frame_number: int

@app.post("/b64")
async def process_image(request: ImageRequest):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image_base64)
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    future = executor.submit(image_handler, image_data, request.pred_conf_limit, request.tiles, request.overlap_percentage, request.frame_number)
    result = future.result()
    # Pass the decoded image and additional parameters to the image handler
    #result = image_handler(image_data, request.pred_conf_limit, request.tiles, request.overlap_percentage)


    return result

@app.get("/")
async def root():
    return {"message": "works"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info", workers=4)


