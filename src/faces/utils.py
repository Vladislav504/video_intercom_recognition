import cv2
from settings import SCALE



def scale_image(image, scale=SCALE):
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)


def ready_image(image):
    small_frame = scale_image(image)
    return cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
