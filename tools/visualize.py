# visualize.py

# goes through every image in a directory and attempts to find a face in each
# if a face is found, the picture is shown briefly with a box around the detected face
# if a face is not found, no picture is shown
#
# this is to see what will be fed into the model during training and
# lets us delete an image if the face detector detects the wrong face
#
# use ctrl-c when a false detection is spotted to end the program and then delete 
#
# when running from the command line, optional parameters are:
#
# -p/--path [path to image directory]
## if not defined, the program will prompt user for path
#
# -s/--start [index of image directory]
## allows user to pick up from where they left off after deleting an image
## using ctrl-c will also show how many pictures were gone through in a session
## so that the user can add that number to their previous start value
## can be modified directly in the code, default is 0
#
# -f/--faceonly [True/False]
## instead of showing entire image with a box drawn around the detected face
## only the detected face will be shown

import argparse
import cv2
import numpy as np
from pathlib import Path
import re

# set start here
START_FROM = 0

# set amount of time each picture is shown for (in ms)
WAIT_TIME = 300

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

net = cv2.dnn.readNetFromCaffe(str(Path().resolve().parents[0] / Path('caffe/deploy.prototxt')), str(Path().resolve().parents[0] / Path('caffe/res10_300x300_ssd_iter_140000.caffemodel')))

def detect_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, w, h) = box.astype("int")
            
            return image[y:h, x:w], (x, y, w, h)
        
        return None, None

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", help="path to image directory")
    ap.add_argument("-s", "--start", type=int, help="image number to start from")
    ap.add_argument("-f", "--faceonly", type=bool, help="show face only")
    args = ap.parse_args()

    if args.path:
        path = Path(args.path)
    else:
        path = Path(input("path to image directory: "))

    try:
        if args.start:
            START_FROM = args.start       
        
        # because my image files were named as "handong (1).jpg" and such,
        # the key is used to go through the pictures in order
        pic_dir = sorted(path.iterdir(), key=lambda x: int(re.search(r'(?<=\()\d+(?=\))', str(x))[0]))

        for count, p in enumerate(pic_dir[min(START_FROM, len(pic_dir) - 1):]):
            print("working on:", p)

            try:
                img_array = cv2.imread(str(p))

                if img_array.shape[1] < img_array.shape[0] and img_array.shape[1] > SCREEN_HEIGHT//5*2:
                    img_array = cv2.resize(img_array, (int(SCREEN_HEIGHT//5*2 * img_array.shape[1]/img_array.shape[0]), SCREEN_HEIGHT//5*2))
                elif img_array.shape[0] < img_array.shape[1] and img_array.shape[0] > SCREEN_WIDTH//5*2:
                    img_array = cv2.resize(img_array, (SCREEN_WIDTH//5*2, int(SCREEN_WIDTH//5*2 * img_array.shape[0]/img_array.shape[1])))
                else:
                    img_array = cv2.resize(img_array, (600, int(600 * img_array.shape[0]/img_array.shape[1])))
                
                face, face_coords = detect_face(img_array)

                if not args.faceonly:
                    if face_coords is not None:
                        (x, y, w, h) = face_coords
                        cv2.rectangle(img_array, (x, y), (w, h), (255, 0, 0), 2)
                        
                        cv2.imshow(p.name, img_array)
                        cv2.waitKey(WAIT_TIME)
                        cv2.destroyAllWindows()
                else:
                    if face is not None:
                        cv2.imshow(p.name, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        cv2.waitKey(WAIT_TIME)
                        cv2.destroyAllWindows()    

            except Exception as e:
                print("unable to read image:", p)

    except KeyboardInterrupt:
        print("processed", count, "pictures")
        raise

    print("processed", count, "pictures")
