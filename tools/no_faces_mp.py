import argparse
import cv2
import multiprocessing as mp
import numpy as np
from pathlib import Path


net = cv2.dnn.readNetFromCaffe(str(Path().resolve().parents[0] / Path('caffe/deploy.prototxt')),
                               str(Path().resolve().parents[0] / Path('caffe/res10_300x300_ssd_iter_140000.caffemodel')))


def detect_face(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.65:

            return True
        
    return False

def handle_image(p: Path):
    if p.is_file():
        print(f"{mp.current_process().name:<18}:", p)

        try:
            img = cv2.imread(str(p))
            
            found = detect_face(img)

            if not found:
                print("no face detected in", p, '\ndeleting...')
                p.unlink()

                return 1

            return 0
                
        except Exception as e:
            print("unable to read image", p, '\ndeleting...')
            p.unlink()

            return 1
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", help="path to image directory")
    args = ap.parse_args()

    count = 0

    if args.path:
        path = Path(args.path)

        pic_dir = list(path.iterdir())

        with mp.Pool(processes=mp.cpu_count()) as pool:
            count += sum(pool.map(handle_image, pic_dir))

    print('deleted', count, 'pictures')
