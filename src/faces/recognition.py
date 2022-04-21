from queue import Queue
from typing import Optional
import face_recognition
from threading import Thread
import cv2
import numpy as np
from faces.fetchers import FacesFetcherProtocol
from faces.utils import ready_image
from settings import SCALE



class FaceComparasion:
    """Сравнивает одно лицо с другими и выдает имя, если произошел match"""

    def __init__(self, fetcher: FacesFetcherProtocol) -> None:
        self.fetcher = fetcher

    def get_name(self, face_encoding):
        matches = face_recognition.compare_faces(self.fetcher.encodings, face_encoding)
        face_distances = face_recognition.face_distance(
            self.fetcher.encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return self.fetcher.names[best_match_index]
        return "Unknown"


class FaceRecognitionStream:
    """Поток распознанных лиц из другого потока изображений"""

    def __init__(
        self,
        threads=3,
        max_queue=1024,
        add_name=True,
        faces: Optional[FaceComparasion] = None,
    ):
        self.frames = Queue(maxsize=max_queue)
        self.processed = Queue(maxsize=max_queue)
        self.threads = threads
        self.stop_worker = False
        self.add_name = add_name
        self.faces = faces

    def start(self):
        Thread(target=self.recognized_on_all_frames_worker, args=()).start()
        for _ in range(self.threads):
            Thread(target=self.face_recognizer_worker, args=()).start()
        return self

    def face_recognizer_worker(
        self,
    ):
        while True:
            if self.stop_worker:
                break
            image = self.frames.get()
            image = self.image_pipeline(image)
            self.add_processed(image)

    def image_pipeline(self, image):
        image_converted = ready_image(image)
        faces_on_frame = face_recognition.face_locations(image_converted)
        face_encodings = face_recognition.face_encodings(
            image_converted, faces_on_frame
        )
        self.set_presented_faces(faces_on_frame)
        self.draw_rectangle_on_face(image, faces_on_frame)
        self.set_names_on_faces(
            image, face_locations=faces_on_frame, face_encodings=face_encodings
        )
        return image

    def add_processed(self, image):
        if not self.processed.full():
            self.processed.put(image)

    def set_presented_faces(self, faces):
        if len(faces) > 0:
            self.face_presents_decisions.append(True)
        self.face_presents_decisions.append(False)

    def draw_rectangle_on_face(self, image, faces):
        for y1, x2, y2, x1 in faces:
            scale_up = int(1 / SCALE)
            y1, x2, y2, x1 = (
                y1 * scale_up,
                x2 * scale_up,
                y2 * scale_up,
                x1 * scale_up,
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

        return image

    def set_names_on_faces(self, image, face_locations, face_encodings):
        if not self.add_name or self.faces is None:
            return image

        for (y1, x2, y2, x1), face_encoding in zip(face_locations, face_encodings):
            name = self.faces.get_name(face_encoding)
            scale_up = int(1 / SCALE)
            y1, x2, y2, x1 = (
                y1 * scale_up,
                x2 * scale_up,
                y2 * scale_up,
                x1 * scale_up,
            )
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)
        return image

    def add_frame(self, image):
        if not self.frames.full():
            self.frames.put(image)

    def get_recognized(self):
        return self.processed.get()

    def empty(self):
        return self.processed.empty()

    def stop(self):
        self.stop_worker = True
