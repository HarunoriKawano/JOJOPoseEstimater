import torch
import numpy as np
import cv2
import opencv_functions as cvF

from segmentation import function as seg_F
from movie import MovieCreator


def image_synthesis(target, back, top_x, top_y, alpha):
    height, width = target.shape[:2]
    if back.ndim == 4:
        back = cv2.cvtColor(back, cv2.COLOR_RGBA2RGB)
    if target.ndim == 3:
        target = cv2.cvtColor(target, cv2.COLOR_RGB2RGBA)

    mask = target[:, :, 3]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask / 255 * alpha

    target = target[:, :, :3]
    back = back.astype(np.float64)
    target = target.astype(np.float64)

    back[top_y:height+top_y:, top_x:width+top_x] *= 1 - mask
    back[top_y:height+top_y:, top_x:width+top_x] += target * mask

    return back.astype(np.uint8)


if __name__ == '__main__':
    # Initial values
    image_path = "data/test6.jpg"
    while True:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception

        while True:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                image = frame.copy()
                cap.release()
                cv2.destroyAllWindows()
                break

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'use device: {device}')

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        person_image = seg_F.make_clipped_person(image, height, width, device)

        cv2.imshow("person", cvF.scale_to_height(person_image, 500))
        key = cv2.waitKey(0)
        if key == ord('y'):
            cv2.destroyAllWindows()
            break

    person_image = cvF.scale_to_height(person_image, 1000)
    cv2.imwrite('cutting.png', person_image)

    name = input('Enter your name: ')

    detection_result = 0  # detection result

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('ImgVideo.mp4', fourcc, 30.0, (1920, 1080))

    mv = MovieCreator(video, person_image, detection_result, name, 4, maxparam=False)
    video, last_picture = mv.forward()

    video.release()
    cv2.imwrite('test.png', last_picture)
