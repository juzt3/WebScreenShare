import cv2
import numpy as np
import time
import mss
import threading
import base64
import requests
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from ctypes import windll


def psleep(num: int):
    windll.winmm.timeBeginPeriod(1)
    windll.kernel32.Sleep(int(num))
    windll.winmm.timeEndPeriod(1)


class ScreenStreamer(threading.Thread):
    def __init__(self, scale=0.2, fps_limit=30, monitor=None, url=None, stream_id=None):
        threading.Thread.__init__(self)
        self.scale = scale
        self.fps_limit = fps_limit
        self.monitor = monitor or mss.mss().monitors[0]
        self.top_grab = {
            'left': self.monitor['left'],
            'top': self.monitor['top'],
            'width': self.monitor['width'],
            'height': round(self.monitor['height'] / 2)
        }
        self.down_grab = {
            'left': self.monitor['left'],
            'top': self.monitor['top'] + round(self.monitor['height'] / 2),
            'width': self.monitor['width'],
            'height': round(self.monitor['height'] / 2)
        }
        self.queue_top = Queue(maxsize=1)
        self.queue_down = Queue(maxsize=1)
        self.is_running = True
        self.display = True
        self.singleThread = False
        self.url = url
        self.stream_id = stream_id

    def grab_frame(self, queue, grab_area):
        with mss.mss() as sct:
            while self.is_running:
                last_time = time.time()

                pre_frame = np.array(sct.grab(grab_area))
                pre_frame = cv2.resize(pre_frame, None, fx=self.scale, fy=self.scale,
                                       interpolation=cv2.INTER_NEAREST)
                queue.put(pre_frame)

                try:
                    fps = 1 / (time.time() - last_time)
                    if fps > self.fps_limit:
                        sleep_time = max(0, 1 / self.fps_limit - (time.time() - last_time))
                        psleep(sleep_time*1000)
                except ZeroDivisionError:
                    continue
        print('Grabber Finished!')

    def displayer(self):
        top_frame = None
        down_frame = None
        max_fps = 0
        num_fps = 0
        sum_fps = 0

        while self.is_running:
            last_time = time.time()

            if self.singleThread:
                full_frame = self.queue_top.get_nowait() if not self.queue_top.empty() else top_frame
                if full_frame is not None:
                    cv2.imshow("Frame", full_frame)

                    try:
                        fps = 1 / (time.time() - last_time)
                        if fps > self.fps_limit:
                            sleep_time = max(0, 1 / self.fps_limit - (time.time() - last_time))
                            psleep(sleep_time*1000)
                        fps = 1 / (time.time() - last_time)
                        sum_fps += fps
                        num_fps += 1
                        avg_fps = sum_fps / num_fps
                        max_fps = max(fps, max_fps)
                        cv2.setWindowTitle("Frame", f"Avg FPS: {avg_fps} | Max FPS: {max_fps}")
                    except ZeroDivisionError:
                        continue

            else:
                top_frame = self.queue_top.get_nowait() if not self.queue_top.empty() else top_frame
                down_frame = self.queue_down.get_nowait() if not self.queue_down.empty() else down_frame

                if top_frame is not None and down_frame is not None:
                    full_frame = np.concatenate((top_frame, down_frame), axis=0)
                    cv2.imshow("Frame", full_frame)

                    try:
                        fps = 1 / (time.time() - last_time)
                        if fps > self.fps_limit:
                            sleep_time = max(0, 1 / self.fps_limit - (time.time() - last_time))
                            psleep(sleep_time*1000)
                        fps = 1 / (time.time() - last_time)
                        sum_fps += fps
                        num_fps += 1
                        avg_fps = sum_fps / num_fps
                        max_fps = max(fps, max_fps)
                        cv2.setWindowTitle("Frame", f"Avg FPS: {avg_fps} | Max FPS: {max_fps}")
                    except ZeroDivisionError:
                        continue

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                self.is_running = False

        print('Displayer Finished!')

    def send_frame(self):
        top_frame = None
        down_frame = None

        while self.is_running:
            last_time = time.time()

            if self.singleThread:
                full_frame = self.queue_top.get_nowait() if not self.queue_top.empty() else top_frame
                if full_frame is not None:
                    try:
                        fps = 1 / (time.time() - last_time)
                        if fps > self.fps_limit:
                            sleep_time = max(0, 1 / self.fps_limit - (time.time() - last_time))
                            psleep(sleep_time*1000)
                    except ZeroDivisionError:
                        continue

                    _, buffer = cv2.imencode('.webp', full_frame, [cv2.IMWRITE_WEBP_QUALITY, 75])

                    base64str = base64.b64encode(buffer).decode("utf-8")
                    uri = "http://" + self.url + "/send_frame_from_string/" + self.stream_id
                    json_str = f'{{"img_base64str": "{base64str}"}}'
                    try:
                        requests.post(uri, json_str)
                    except:
                        continue
            else:
                top_frame = self.queue_top.get_nowait() if not self.queue_top.empty() else top_frame
                down_frame = self.queue_down.get_nowait() if not self.queue_down.empty() else down_frame

                if top_frame is not None and down_frame is not None:
                    try:
                        fps = 1 / (time.time() - last_time)
                        if fps > self.fps_limit:
                            sleep_time = max(0, 1 / self.fps_limit - (time.time() - last_time))
                            psleep(sleep_time*1000)
                    except ZeroDivisionError:
                        continue
                    full_frame = np.concatenate((top_frame, down_frame), axis=0)
                    _, buffer = cv2.imencode('.webp', full_frame, [cv2.IMWRITE_WEBP_QUALITY, 75])
                    base64str = base64.b64encode(buffer).decode("utf-8")
                    uri = "http://"+self.url+"/send_frame_from_string/"+self.stream_id
                    json_str = f'{{"img_base64str": "{base64str}"}}'
                    try:
                        requests.post(uri, json_str)
                    except:
                        continue

    def run(self):
        if self.singleThread:
            with ThreadPoolExecutor(max_workers=3) as executor:
                t1 = executor.submit(self.grab_frame, self.queue_top, self.monitor)
                if self.display:
                    t2 = executor.submit(self.displayer)
                    t2.result()
                if self.url is not None and self.stream_id is not None:
                    t3 = executor.submit(self.send_frame)
                    t3.result()
                t1.result()
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                t1 = executor.submit(self.grab_frame, self.queue_top, self.top_grab)
                t2 = executor.submit(self.grab_frame, self.queue_down, self.down_grab)
                if self.display:
                    t3 = executor.submit(self.displayer)
                    t3.result()
                if self.url is not None and self.stream_id is not None:
                    t4 = executor.submit(self.send_frame)
                    t4.result()
                t1.result()
                t2.result()

    def stop(self):
        self.is_running = False
        print('All threads terminated!')


if __name__ == "__main__":
    from WindowFinder import getWindowMonitor
    monitor = getWindowMonitor("Albion Online Client")
    streamer = ScreenStreamer(scale=0.5, fps_limit=15, monitor=monitor)
    streamer.display = True
    streamer.singleProcess = True
    streamer.start()
    while streamer.is_running:
        psleep(1000)
