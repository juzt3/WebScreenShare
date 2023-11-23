from pydantic import BaseModel
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi_frame_stream import FrameStreamer
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio
import mss
from ScreenStreamerThreaded import ScreenStreamer
from WindowFinder import getWindowMonitor

# monitor = getWindowMonitor("Albion Online Client")
monitor = mss.mss().monitors[1]
url = "127.0.0.1:8000"
stream_id = "test_stream"
fps = 5
streamer = ScreenStreamer(scale=0.5, fps_limit=fps, url=url, stream_id=stream_id, monitor=monitor)
streamer.display = False
streamer.singleThread = False

fs = FrameStreamer()


app = FastAPI()
templates = Jinja2Templates(directory="templates")


class InputImg(BaseModel):
    img_base64str: str


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


@app.post("/send_frame_from_string/{stream_id}")
async def send_frame_from_string(stream_id: str, d: InputImg):
    await fs.send_frame(stream_id, d.img_base64str)


@app.post("/send_frame_from_file/{stream_id}")
async def send_frame_from_file(stream_id: str, file: UploadFile = File(...)):
    await fs.send_frame(stream_id, file)


@app.get("/video_feed/{stream_id}")
async def video_feed(stream_id: str):
    return fs.get_stream(stream_id, freq=fps)


if __name__ == "__main__":
    streamer.start()
    config = Config()
    config.bind = ["0.0.0.0:8000"]

    asyncio.run(serve(app, config))
