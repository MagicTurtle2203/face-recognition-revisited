import argparse
import cv2
import numpy as np
from pathlib import Path

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

            return True
        
    return False
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", help="path to image directory")
    args = ap.parse_args()

    if args.path:
        path = Path(args.path)
    else:
        path = Path(input("path to image directory: "))

    count = 0

    pic_dir = list(path.iterdir())

    for p in pic_dir:
        if p.is_file():
            print("working on:", p)

            try:
                img = cv2.imread(str(p))
                
                found = detect_face(img)

                if found is False:
                    print("no face detected in:", p, '\ndeleting...')
                    p.unlink()
                    count += 1
                    
            except Exception as e:
                print("unable to read image:", p, '\ndeleting...')
                p.unlink()
                count += 1

    print('deleted', count, 'pictures')
