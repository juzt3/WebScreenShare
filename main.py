import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi_frame_stream import FrameStreamer

from ScreenStreamer import ScreenStreamer

url = "127.0.0.1:8888"
stream_id = "test_stream"
streamer = ScreenStreamer(scale=0.25, fps_limit=15, url=url, stream_id=stream_id)
streamer.display = False

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
    return fs.get_stream(stream_id)


if __name__ == "__main__":
    streamer.start()
    uvicorn.run(
        "main:app",
        port=8888,
        reload=True,
        log_level="debug",
        workers=2
    )
