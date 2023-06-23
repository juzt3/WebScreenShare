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
        self.monitor = monitor or mss.mss().monitors[0]
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
        self.display = True
        self.url = url
        self.stream_id = stream_id

    def grab_frame(self, queue, grab_area):
        with mss.mss() as sct:
            while self.is_running.value:
                pre_frame = np.array(sct.grab(grab_area))
                pre_frame = cv2.resize(pre_frame, None, fx=self.scale, fy=self.scale,
                                       interpolation=cv2.INTER_NEAREST)
                queue.put(pre_frame)
        print('Grabber Finished!')

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

    def send_frame(self):
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
        p1 = Process(target=self.grab_frame, args=(self.queue_top, self.top_grab))
        p2 = Process(target=self.grab_frame, args=(self.queue_down, self.down_grab))
        p1.start()
        p2.start()
        p3 = None
        if self.display:
            p3 = Process(target=self.displayer)
        elif self.url is not None and self.stream_id is not None:
            p3 = Process(target=self.send_frame)
        else:
            raise Exception("The 'display' option or the 'url' and 'stream_id' should be set.")

        if p3:
            p3.start()
        while self.is_running.value:
            time.sleep(1)
        self.stop()
        p1.terminate()
        p2.terminate()
        if p3:
            p3.terminate()

    def stop(self):
        self.is_running.value = 0
        print('All processes terminated!')


if __name__ == "__main__":
    streamer = ScreenStreamer(scale=0.5, fps_limit=60)
    streamer.display = True
    streamer.start()
    while streamer.is_running.value:
        time.sleep(1)
