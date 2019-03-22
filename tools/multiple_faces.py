import argparse
import cv2
import numpy as np
from pathlib import Path

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

net = cv2.dnn.readNetFromCaffe(str(Path().resolve().parents[0] / Path('caffe/deploy.prototxt')), str(Path().resolve().parents[0] / Path('caffe/res10_300x300_ssd_iter_140000.caffemodel')))

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
            
                    if i_area / ((first[2] - first[0]) * (first[3] - first[1])) < 0.05:
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
            img = cv2.imread(str(p))
            
            if img.shape[1] < img.shape[0]:
                img = cv2.resize(img, (int(SCREEN_HEIGHT//2 * img.shape[1]/img.shape[0]),
                                       SCREEN_HEIGHT//2))
            elif img.shape[0] < img.shape[1]:
                img = cv2.resize(img, (SCREEN_WIDTH//5*2,
                                           int(SCREEN_WIDTH//5*2 * img.shape[0]/img.shape[1])))
            if img.shape[1] < img.shape[0]:
                x_shift = img.shape[1]//10
                y_shift = 0
                cut_img = img[0:img.shape[0]*6//7, x_shift:x_shift*9]
            else:
                x_shift = img.shape[1]//8
                y_shift = img.shape[0]//10
                cut_img = img[y_shift:y_shift*9, x_shift:x_shift*7]

            face_list = detect_face(cut_img)
            cut = True

            if len(face_list) == 0:
                face_list = detect_face(img)
                cut = False

            if len(face_list) > 1:

                for n, (x, y, w, h) in enumerate(face_list):
                    if cut is True:
                        x, w = x+x_shift, w+x_shift
                        y, h = y+y_shift, h+y_shift
                    
                    cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)
                    cv2.putText(img, str(n), (x, y - 10 if y - 10 > 10 else y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                cv2.imshow(p.name, img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                warning = input("would you like to delete (y/n)? ")

                if warning in ['yes', 'y', '']:
                    print("deleting:", p)
                    p.unlink()

        except Exception as e:
            print("deleting:", p)
            p.unlink()
