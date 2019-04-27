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
#   if not defined, the program will prompt user for path
#
# -s/--start [index of image directory]
#   allows user to pick up from where they left off after deleting an image
#   using ctrl-c will also show how many pictures were gone through in a session
#   so that the user can add that number to their previous start value
#   can be modified directly in the code, default is 0
#
# -d/--debug [True]
#   will show all faces detected (with selected face shown in blue and others
#   shown in red) along with the confidence level for each face as well as 
#   print out some information about each detection such as its shape and distance
#   from the center of the image

import argparse
import cv2
import math
import numpy as np
import re
from pathlib import Path

# set start here
START_FROM = 0

# det debug here
DEBUG = False

# set amount of time each picture is shown for (in ms)
WAIT_TIME = 300

# change for screen resolution in order to make all pictures fit the screen
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


net = cv2.dnn.readNetFromCaffe(str(Path().resolve().parents[0] / Path('caffe/deploy.prototxt')),
                               str(Path().resolve().parents[0] / Path('caffe/res10_300x300_ssd_iter_140000.caffemodel')))


def get_center(size: tuple) -> tuple:
    x, y, w, h = size
    center_x = (w + x) // 2
    center_y = (h + y) // 2
    return center_x, center_y


def distance(center_1: tuple, center_2: tuple) -> float:
    return math.sqrt(math.pow(center_1[0] - center_2[0], 2) +
                     math.pow(center_1[1] - center_2[1], 2))


def distance_x(center_1: tuple, center_2: tuple) -> float:
    return abs(center_1[0] - center_2[0])


def detect_face(image, debug: bool = False) -> tuple:
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    detections_set = set()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.65:

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x, y, w, h = box.astype("int")
            
            detections_set.add((confidence, (x, y, w, h)))

    if debug:
        return detections_set
    
    try:
        max_confidence = max(detections_set, key=lambda x: x[0])
        min_distance = min(detections_set, key=lambda x: distance_x(get_center(x[1]), get_center((0, 0, width, height))))
    except ValueError:
        return None
    else:
        if max_confidence[0] - 0.10 > min_distance[0]:
            output = max_confidence[1]
        else:
            output = min_distance[1]
        return output


def debug_draw_rects(image, detections) -> "image":
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    max_confidence = max(detections, key=lambda x: x[0])
    min_distance = min(detections, key=lambda x: distance_x(get_center(x[1]), get_center((0, 0, width, height))))

    if max_confidence[0] - 0.10 > min_distance[0]:
        output = max_confidence[1]
    else:
        output = min_distance[1]

    for confidence, detection in detections:
        print(f"center of image: {get_center((0, 0, width, height))}\n"
            f"detection shape: {detection}\n"
            f"distance from center y-axis: {distance_x(get_center(detection), get_center((0, 0, width, height)))}\n"
            f"confidence: {confidence}\n")

        x, y, w, h = detection

        if detection == output:
            cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 2)
            cv2.putText(image, f"{confidence*100:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
        else:
            cv2.rectangle(image, (x, y), (w, h), (255, 0, 0), 2)
            cv2.putText(image, f"{confidence*100:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2)     
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", help="path to image directory")
    ap.add_argument("-s", "--start", type=int, help="image number to start from")
    ap.add_argument("-d", "--debug", type=bool, help="show all detections")
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

                if DEBUG or args.debug:
                    detections = detect_face(img, True)

                    if detections:
                        img = debug_draw_rects(img, detections)
                        
                        cv2.imshow(p.name, img)
                        if cv2.waitKey(WAIT_TIME) & 0xFF == ord("q"):
                            cv2.destroyAllWindows()
                            break
                        cv2.destroyAllWindows()

                else:
                    coords = detect_face(img)

                    if coords is not None:
                        (x, y, w, h) = coords
                            
                        cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)
                        
                        cv2.imshow(p.name, img)
                        if cv2.waitKey(WAIT_TIME) & 0xFF == ord("q"):
                            cv2.destroyAllWindows()
                            break
                        cv2.destroyAllWindows()

                count += 1

            except Exception as e:
                print(e)
                print("unable to read image:", p)

    except KeyboardInterrupt:
        print("processed", count, "pictures")
        raise

    print("processed", count, "pictures")
