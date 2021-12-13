import os
import cv2
import numpy as np
import argparse

import presenteragent
from acllite_image import AclLiteImage
from acllite_resource import AclLiteReso

PRESENTER_CONF="presenter.conf"

def main(video_path):
    cap = cv2.VideoCapture(frames_input_src)
    chan = presenteragent.presenter_channel.open_channel(BODYPOSE_CONF)
    if chan is None:
        print("Open presenter channel failed")
        return

    while(cap.isOpened()):
        ## Read one frame of the input video ## 
        ret, img_original = cap.read()

        if not ret:
            print('Cannot read more, Reach the end of video')
            break
        
        ## Present Result ##
        if is_presenter_server:
            # convert to jpeg image for presenter server display
            _, jpeg_image = cv2.imencode('.jpg', img_original)
            # construct AclLiteImage object for presenter server
            jpeg_image = AclLiteImage(jpeg_image, img_original.shape[0], img_original.shape[1], jpeg_image.size)
            # send to presenter server
            chan.send_detection_data(img_original.shape[0], img_original.shape[1], jpeg_image, [])
    
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_path', type=str, default=None, help="Directory path for video.")
    args = parser.parse_args()


main(args.input_video_path)