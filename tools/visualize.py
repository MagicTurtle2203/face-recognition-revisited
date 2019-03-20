# visualize.py

# goes through every image in a directory and attempts to find a face in each
# if a face is found, the picture is shown briefly with a box around the detected face
# if a face is not found, no picture is shown
#
# this is to see what will be fed into the model during training and
# lets us delete an image if the face detector detects the wrong face
#
# pause the command line or use ctrl-c to stop the program
# when a false detection is spotted and then delete the picture
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

import argparse
import cv2
import numpy as np
import re
from pathlib import Path

# set start here
START_FROM = 0

# set amount of time each picture is shown for (in ms)
WAIT_TIME = 300

# change for screen resolution in order to make all pictures fit the screen
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

net = cv2.dnn.readNetFromCaffe(str(Path().resolve().parents[0] / Path('caffe/deploy.prototxt')),
                               str(Path().resolve().parents[0] / Path('caffe/res10_300x300_ssd_iter_140000.caffemodel')))

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
            
            return (x, y, w, h)
        
        return None

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", help="path to image directory")
    ap.add_argument("-s", "--start", type=int, help="image number to start from")
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
        pic_dir = sorted(path.iterdir(),
                         key=lambda x: int(re.search(r'(?<=\()\d+(?=\))', str(x))[0]))

        count = 0
        
        for p in pic_dir[min(START_FROM, len(pic_dir) - 1):]:
            print("working on:", p)

            try:
                img = cv2.imread(str(p))

                if img.shape[1] <= img.shape[0]:
                    img = cv2.resize(img, (int(SCREEN_HEIGHT//2 * img.shape[1]/img.shape[0]),
                                           SCREEN_HEIGHT//2))
                else:
                    img = cv2.resize(img, (SCREEN_WIDTH//5*2,
                                           int(SCREEN_WIDTH//5*2 * img.shape[0]/img.shape[1])))

                if img.shape[1] <= img.shape[0]:
                    x_shift = img.shape[1]//8
                    y_shift = 0
                    cut_img = img[0:img.shape[0]*4//5, x_shift:x_shift*7]
                else:
                    x_shift = img.shape[1]//7
                    y_shift = img.shape[0]//10
                    cut_img = img[y_shift:y_shift*9, x_shift:x_shift*6]
                
                cut_coords = detect_face(cut_img)
                
                if cut_coords is not None:
                    cut_x, cut_y, cut_w, cut_h = cut_coords
                    cut_x, cut_w = cut_x+x_shift, cut_w+x_shift
                    cut_y, cut_h = cut_y+y_shift, cut_h+y_shift

                if (cut_x <= 0
                    or cut_y <= 0
                    or cut_w >= img.shape[1]
                    or cut_h >= img.shape[0]):
                    coords = detect_face(img)
                else:
                    coords = (cut_x, cut_y, cut_w, cut_h)

                if coords is not None:
                    (x, y, w, h) = coords
                        
                    cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)
                    
                    cv2.imshow(p.name, img)
                    cv2.waitKey(WAIT_TIME)
                    cv2.destroyAllWindows()

                count += 1

            except Exception as e:
                print("unable to read image:", p)

    except KeyboardInterrupt:
        print("processed", count, "pictures")
        raise

    print("processed", count, "pictures")
