import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from .config import AppConfig
from .pipeline import analyze_file
from .utils import ensure_dir, make_temp_filename

APP_CFG = AppConfig()

UPLOAD_DIR = "uploads"
ensure_dir(UPLOAD_DIR)

app = FastAPI(
    title="Synthetic Speech Fragment Detector",
    version="0.1.0"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не задан")

    ctype = (file.content_type or "").lower()
    allowed = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"}

    if ctype not in allowed:
        raise HTTPException(status_code=415, detail="Неподдерживаемый тип файла")

    tmp_name = make_temp_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, tmp_name)

    t0 = time.time()

    with open(path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    try:
        result = analyze_file(path, file.filename, APP_CFG)
        result["runtime_sec"] = round(time.time() - t0, 4)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

