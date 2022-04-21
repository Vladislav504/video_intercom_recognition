from typing import List, Protocol
import requests
import settings
import glob
import face_recognition
import cv2
from faces.utils import ready_image


class FacesFetcherProtocol(Protocol):
    names: List
    encodings: List

    def load_faces(self, path):
        ...


class ApiFacesFetcher:
    """Получает encodings лиц из папки"""

    names = []
    encodings = []

    def load_faces(self, path="/faces/"):
        faces = requests.get(settings.API_DOMAIN + path)
        faces.raise_for_status()
        faces_data = faces.json()
        assert isinstance(faces_data, dict)
        for name, encoding in faces_data.items():
            self.names.append(name)
            self.encodings.append(encoding)
        print(self.names)


class FilesFacesFetcher:
    """Получает encodings лиц из API"""

    names = []
    encodings = []

    def load_faces(self, path="./data"):
        for image_name in glob.glob(f"{path}/*.jpg"):
            name = image_name[image_name.rfind("/") + 1 : image_name.rfind(".")]
            image = cv2.imread(image_name)
            image_converted = ready_image(image)
            image_encodings = face_recognition.face_encodings(image_converted)
            if len(image_encodings) > 0:
                self.names.append(name)
                self.encodings.append(image_encodings[0])
