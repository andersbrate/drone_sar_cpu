import cv2
import json
import requests
import numpy as np
import base64
import time
import sys
import io
sys.path.insert(1, '/home/POLITIET/abr063/git/drone_sar/')
from tiller_light import tiller_light
from PIL import Image
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

def post_image(video_path):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize frame count
    frame_count = 0
    frame_times = []
    tile_times = []

    while True:
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            print('issue with frame read')
            break


        fig, ax = plt.subplots(1)
        ax.imshow(np.array(frame, dtype=np.uint8))
        _h,_w,_c = np.shape(np.array(frame,dtype=np.uint8)) 
        try:
            str_img = frame_to_base64(frame)
            
            headers = {'content-type': 'application/json', 'Accept': 'text/plain'}
            payload = json.dumps({'image': str_img})
            response = requests.post('http://localhost:5000/b64', data=payload, headers=headers)
            content = response.json()
            #print(response.content)
            if response.status_code == 200:
                print(f"Frame {frame_count} sent successfully.")
                if content['msg'] == 'success':
                    for box, conf in zip(content['boxes'], content['confidence']):
                        print(conf)
                        print(box)
                        if float(conf) > 0.6:
                            x,y,w,h = box[0]*_w,box[1]*_h,box[2]*_w,box[3]*_h
                            ax.add_patch(patches.Rectangle((x-(w/2), y-(h/2)), w, h, linewidth = 1.5, edgecolor = 'r', facecolor='none'))
                    
                    

                
            else:
                print(f"Error sending frame {frame_count}. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error sending frame {frame_count}: {e}")
        
        fig.savefig('temp_videofile/'+str(frame_count)+'_.jpg')
        plt.close(fig)
        # Increment frame count
        frame_count += 1

    # Release the video capture
    cap.release()
    return True


if __name__ == '__main__':
    video_path = '../skookumCreek_nobox.mp4'
    frame_times = post_image(video_path)
