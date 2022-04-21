from queue import Queue
import json
from threading import Thread
from websocket import create_connection


class WebSocketStreamProtocol:
    def add(self, message):
        ...


class WebSocketStream:
    """Отправка потока данных в вебсокет"""

    def __init__(self, url: str) -> None:
        self.ws = None
        self.url = url
        self.buffer = Queue(maxsize=20)
        self.running = True

    def start(self):
        self.ws = create_connection(self.url)
        Thread(target=self.send, args=()).start()
        return self

    def send(self):
        while self.running:
            if not self.buffer.empty():
                if not self.ws.connected:
                    self.ws = create_connection(self.url)
                self.ws.send(json.dumps(self.buffer.get()))

    def add(self, message):
        if not self.buffer.full():
            self.buffer.put(message)

    def close(self):
        self.running = False
        self.ws.close()
