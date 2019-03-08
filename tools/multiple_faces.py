import argparse
import cv2
import numpy as np
from pathlib import Path

net = cv2.dnn.readNetFromCaffe(str(Path('./caffe/deploy.prototxt')), str(Path('./caffe/res10_300x300_ssd_iter_140000.caffemodel')))

def detect_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    detection_list = []

    first = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, w, h) = box.astype("int")

            if first is None:
                detection_list.append((x, y, w, h))
                first = (x, y, w, h)

            else:
                if w - x > (first[2] - first[0]) * 0.5 and h - y > (first[3] - first[1]) * 0.5:     
                    i_area = max(0, (min(w, first[2]) - max(x, first[0]))) * max(0, (min(h, first[3]) - max(y, first[1])))

                    if i_area / ((first[2] - first[0]) * (first[3] - first[1])) < 0.1:
                        detection_list.append((x, y, w, h))
                

    return detection_list

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", help="path to image directory")
    args = ap.parse_args()

    if args.path:
        path = Path(args.path)
    else:
        path = Path(input("path to image directory: "))

    for p in path.iterdir():
        print("working on:", p)
        try:
            img_array = cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (0,0), fx=0.75, fy=0.75)
            face_list = detect_face(img_array)
            if len(face_list) > 1:
                for n, (x, y, w, h) in enumerate(face_list):
                    cv2.rectangle(img_array, (x, y), (w, h), (0, 0, 255), 2)
                    cv2.putText(img_array, str(n), (x, y - 10 if y - 10 > 10 else y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(p.name, cv2.cvtColor(cv2.resize(img_array, (0,0), fx=0.4, fy=0.4), cv2.COLOR_BGR2RGB))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                warning = input("would you like to delete (y/n)? ")
                if warning in ['yes', 'y', '']:
                    print("deleting:", p)
                    p.unlink()
        except Exception as e:
            print("deleting:", p)
            p.unlink()
