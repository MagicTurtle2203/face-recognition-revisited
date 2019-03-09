import argparse
import cv2
import numpy as np
from pathlib import Path

net = cv2.dnn.readNetFromCaffe(str(Path().resolve().parents[0] / Path('caffe/deploy.prototxt')), str(Path().resolve().parents[0] / Path('caffe/res10_300x300_ssd_iter_140000.caffemodel')))

def detect_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, w, h) = box.astype("int")
            
            return image[y:h, x:w], (x, y, w, h)
        
        return None, None

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", help="path to image directory")
    ap.add_argument("-f", "--faceonly", type=bool, help="show face only")
    args = ap.parse_args()

    if args.path:
        path = Path(args.path)
    else:
        path = Path(input("path to image directory: "))
    
    picture_count = 0

    try:
        start_from = 0
        
        pic_dir = list(path.iterdir())
        for p in pic_dir[min(start_from, len(pic_dir) - 1):]:
            print("working on:", p)
            try:
                img_array = cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (0,0), fx=0.75, fy=0.75)
                face, face_coords = detect_face(img_array)
                if not args.faceonly:
                    if face_coords is not None:
                        (x, y, w, h) = face_coords
                        cv2.rectangle(img_array, (x, y), (w, h), (0, 0, 255), 2)
                        cv2.putText(img_array, '0', (x, y - 10 if y - 10 > 10 else y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow(p.name, cv2.cvtColor(cv2.resize(img_array, (0,0), fx=0.4, fy=0.4), cv2.COLOR_BGR2RGB))
                        cv2.waitKey(300)
                        cv2.destroyAllWindows()
                else:
                    if face is not None:
                        cv2.imshow(p.name, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        cv2.waitKey(300)
                        cv2.destroyAllWindows()
                picture_count += 1
            except Exception as e:
                print("unable to read image:", p)
    except KeyboardInterrupt:
        print("processed", picture_count, "pictures")
        raise