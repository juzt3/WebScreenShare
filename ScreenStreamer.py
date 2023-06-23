import cv2
import numpy as np
import time
import mss
from multiprocessing import Process, Queue, Value
import base64
import requests


class ScreenStreamer:
    def __init__(self, scale=0.2, fps_limit=30, monitor=None, url=None, stream_id=None):
        self.scale = scale
        self.fps_limit = fps_limit
        self.monitor = mss.mss().monitors[0] if monitor is None else monitor
        self.top_grab = {
            'left': 0,
            'top': 0,
            'width': self.monitor['width'],
            'height': round(self.monitor['height'] / 2)
        }
        self.down_grab = {
            'left': 0,
            'top': round(self.monitor['height'] / 2),
            'width': self.monitor['width'],
            'height': round(self.monitor['height'] / 2)
        }
        self.queue_top = Queue(maxsize=3)
        self.queue_down = Queue(maxsize=3)
        self.is_running = Value('i', 1)
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.display = True
        self.url = url
        self.stream_id = stream_id

    def grabber_top(self):
        with mss.mss() as sct:
            while self.is_running.value:
                pre_frame = np.array(sct.grab(self.top_grab))
                pre_frame = cv2.resize(
                    pre_frame,
                    dsize=(
                        round(self.top_grab['width'] * self.scale),
                        round(self.top_grab['height'] * self.scale)
                    ),
                    interpolation=cv2.INTER_NEAREST
                )
                self.queue_top.put(pre_frame)
        print('Top Grabber Finished!')

    def grabber_down(self):
        with mss.mss() as sct:
            while self.is_running.value:
                pre_frame = np.array(sct.grab(self.down_grab))
                pre_frame = cv2.resize(
                    pre_frame,
                    dsize=(
                        round(self.down_grab['width'] * self.scale),
                        round(self.down_grab['height'] * self.scale)
                    ),
                    interpolation=cv2.INTER_NEAREST
                )
                self.queue_down.put(pre_frame)
        print('Down Grabber Finished!')

    def displayer(self):
        top_frame = None
        down_frame = None
        max_fps = 0
        num_fps = 0
        sum_fps = 0

        while self.is_running.value:
            last_time = time.time()

            top_frame = self.queue_top.get_nowait() if not self.queue_top.empty() else top_frame
            down_frame = self.queue_down.get_nowait() if not self.queue_down.empty() else down_frame

            if top_frame is not None and down_frame is not None:
                full_frame = np.concatenate((top_frame, down_frame), axis=0)
                cv2.imshow("OpenCV/Numpy normal", full_frame)

                try:
                    fps = 1 / (time.time() - last_time)
                    while fps > self.fps_limit:
                        fps = 1 / (time.time() - last_time)
                        time.sleep(0.001)
                    sum_fps += fps
                    num_fps += 1
                    avg_fps = sum_fps / num_fps
                    print(f"Avg Fps: {avg_fps}")
                    max_fps = max(fps, max_fps)
                    print(f"Max Fps: {max_fps}")
                except ZeroDivisionError:
                    continue

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                self.is_running.value = 0

        print('Displayer Finished!')

    def getFrame(self, asBytes=True):
        self.display = False
        self.start()
        top_frame = None
        down_frame = None
        frame = None

        top_frame = self.queue_top.get()
        down_frame = self.queue_down.get()

        frame = np.concatenate((top_frame, down_frame), axis=0)
        if asBytes:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        self.stop()
        return frame

    def streamer(self):
        top_frame = None
        down_frame = None

        while self.is_running.value:
            top_frame = self.queue_top.get_nowait() if not self.queue_top.empty() else top_frame
            down_frame = self.queue_down.get_nowait() if not self.queue_down.empty() else down_frame

            if top_frame is not None and down_frame is not None:
                full_frame = np.concatenate((top_frame, down_frame), axis=0)
                _, buffer = cv2.imencode('.jpg', full_frame)
                base64str = base64.b64encode(buffer).decode("utf-8")
                uri = "http://"+self.url+"/send_frame_from_string/"+self.stream_id
                json_str = f'{{"img_base64str": "{base64str}"}}'
                try:
                    requests.post(uri, json_str)
                except ConnectionError:
                    continue

    def start(self):
        p1 = Process(target=self.grabber_top)
        p2 = Process(target=self.grabber_down)
        p1.start()
        p2.start()
        if self.display:
            p3 = Process(target=self.displayer)
            p3.start()
        elif self.url is not None and self.stream_id is not None:
            p3 = Process(target=self.streamer)
            p3.start()
        else:
            raise Exception("display option or Url and stream id should be set.")

    def stop(self):
        self.is_running.value = 0
        if self.p1:
            self.p1.terminate()
        if self.p2:
            self.p2.terminate()
        if self.p3:
            self.p3.terminate()
        print('All processes terminated!')


if __name__ == "__main__":
    streamer = ScreenStreamer(scale=1, fps_limit=999)
    streamer.display = True
    streamer.start()
    while streamer.is_running.value:
        time.sleep(1)
    streamer.stop()
