import cv2
import json
import requests
import base64
import time
import sys
import io
from shapely.geometry import Polygon
from itertools import combinations
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def frame_to_base64(image):
    # Convert the NumPy array to PIL Image
    pil_image = Image.fromarray(image)
    # Create an in-memory buffer
    buffer = io.BytesIO()
    # Save the PIL Image to the buffer in PNG format
    pil_image.save(buffer, format='PNG')
    # Seek the buffer to the beginning
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return base64_image


def post_img(image, pred_conf_limit, tiles, overlap_percentage, frame_number):
    str_img = frame_to_base64(image)
    
    headers = {'content-type': 'application/json', 'Accept': 'text/plain'}
    payload = json.dumps({'image_base64': str_img, "pred_conf_limit": pred_conf_limit, "tiles": tiles, "overlap_percentage": overlap_percentage, "frame_number": frame_number})
    response = requests.post('http://localhost:999/b64', data=payload, headers=headers)
    content = response.json()
    detections = content['detections']
    confidence = content['confidence']
    return detections, confidence, response

def post_image(video_path, pred_conf_limit, tiles, overlap_percentage):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize frame count
    frame_count = 0
    times = []

    while True:
        # Read a frame from the video
        success, _frame = cap.read()
        if not success:
            print('issue with frame read')
            break
        detections = [] 
        scores = []

        #fig, ax = plt.subplots(4,3)
        #axs = ax.flatten()
        #for i,x in enumerate(frames):
        #    axs[i].imshow(x)
        #plt.show()
        
        init_time = time.time()
        try:
            detections, scores, response = post_img(_frame, pred_conf_limit, tiles, overlap_percentage, frame_count)
            content = response.json()
            if response.status_code == 200:
                print(f"Frame {frame_count} sent successfully.")
                print(f' results; {content["message"]}, \n {content["detections"]} \n {content["confidence"]}')

        except requests.RequestException as e:
            print(f"Error sending frame {frame_count}: {e}")
        times.append(time.time() - init_time)

        """
        _fig, _ax = plt.subplots(1)
        _ax.imshow(np.array(_frame))
        for elem in detections:
            print('for elem in detections')
            print(elem)
            _ax.add_patch(patches.Rectangle((elem[0], elem[1]), elem[2], -elem[3], linewidth = 1.5, edgecolor = 'r', facecolor='none'))

        _fig.savefig(f'temp_videofile/non_nms_video/{frame_count}.png')

        plt.close()
        """

        # Increment frame count
        frame_count += 1
        #if frame_count >= 10:
        #    return True, times

    # Release the video capture
    cap.release()
    return True, times


if __name__ == '__main__':
    pred_conf_limit = 0.7
    overlap_percentage = 0.2
    tiles = (3,2)
    video_path = 'carnation_enterprise_ir.mp4'

    bool_res, times = post_image(video_path, pred_conf_limit, tiles, overlap_percentage)


    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = cap.read()
        respons
    """

