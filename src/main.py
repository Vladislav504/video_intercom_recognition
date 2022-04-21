import base64
from time import sleep
import cv2

from settings import SOCKET_SCALE, WS_DOMAIN
from speech.speech import VoiceStream
from webcam import WebcamVideoStream
from transport import WebSocketStream, WebSocketStreamProtocol
from faces.recognition import FaceRecognitionStream, FaceComparasion
from faces.fetchers import ApiFacesFetcher
from faces.utils import scale_image


def encode_image(image):
    _, im_buf_arr = cv2.imencode(".jpg", image)
    # decode нужен чтобы получить валидную строку
    return base64.b64encode(im_buf_arr).decode()


def send_to_socket(message, socket: WebSocketStreamProtocol):
    image_message = {"message": message}
    socket.add(image_message)


if __name__ == "__main__":
    faces_fetcher = ApiFacesFetcher()
    faces_fetcher.load_faces()
    faces = FaceComparasion(faces_fetcher)
    socket_image_stream = WebSocketStream(f"{WS_DOMAIN}/ws/image/", 30).start()
    socket_text_stream = WebSocketStream(f"{WS_DOMAIN}/ws/text/", 2).start()
    recognizer = FaceRecognitionStream(threads=5, faces=faces).start()
    vs = WebcamVideoStream(src=0).start()
    voice = VoiceStream().start()
    sleep(1)
    while True:
        image = vs.read()
        recognizer.add_frame(image)
        image = recognizer.get_recognized()
        small_image = scale_image(image, scale=SOCKET_SCALE)
        encoded_image = encode_image(small_image)
        send_to_socket(encoded_image, socket_image_stream)
        if not voice.text.empty():
            text_message = voice.text.get()
            send_to_socket(text_message, socket_text_stream)
