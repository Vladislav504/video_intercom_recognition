from queue import Queue
from time import sleep
import speech_recognition as sr
from threading import Thread
import speech.microphone as microphone


class VoiceStream:
    """Поток распознаного голоса"""

    def __init__(self):
        self.text = Queue(maxsize=5)

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.recognize, args=()).start()
        return self

    def recognize(self):
        microphone.start_microphone(self.text)
