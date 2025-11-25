# agrix/backend/app/main.py
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid, time

from .inference import infer_bytes
from .model_utils import save_result, IMAGES

FE_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Agrix API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend from / (optional)
if FE_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FE_DIR), html=True), name="frontend")

def background_process(image_path: str, sensors: dict, job_id: str):
    time.sleep(0.5)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    out = infer_bytes(image_bytes)
    # create advisory using simple rules or lookup (extend this)
    advisory = f"Suspected {out['disease']}. Follow local agronomy guidelines."
    out_payload = {
        "disease": out["disease"],
        "confidence": out["confidence"],
        "severity": "moderate" if out["confidence"] < 0.85 else "high",
        "advisory": advisory,
        "saliency_url": None
    }
    save_result(job_id, out_payload)

@app.post("/api/v1/upload")
async def upload(background_tasks: BackgroundTasks,
                 image: UploadFile = File(...),
                 rh24: float = Form(None),
                 temp48: float = Form(None),
                 soil_moisture: float = Form(None),
                 recent_rain: int = Form(0)):
    job_id = str(uuid.uuid4())
    ext = Path(image.filename).suffix or ".jpg"
    img_name = f"{job_id}{ext}"
    img_path = IMAGES / img_name
    with img_path.open("wb") as f:
        f.write(await image.read())

    sensors = {"rh24":rh24, "temp48":temp48, "soil_moisture":soil_moisture, "recent_rain":recent_rain}
    background_tasks.add_task(background_process, str(img_path), sensors, job_id)
    return JSONResponse({"job_id": job_id, "status": "processing"})

@app.get("/api/v1/result/{job_id}")
def get_result(job_id: str):
    f = Path(IMAGES).parents[0] / "results" / f"{job_id}.json"
    if not f.exists():
        return JSONResponse({"status":"processing"}, status_code=202)
    return JSONResponse(f.read_text() and __import__("json").loads(f.read_text()))

@app.post("/api/v1/feedback/{job_id}")
async def feedback(job_id: str, payload: dict):
    fb_dir = Path(IMAGES).parents[0] / "feedback"
    fb_dir.mkdir(exist_ok=True)
    fb_file = fb_dir / f"{job_id}.feedback.json"
    fb_file.write_text(__import__("json").dumps(payload))
    return JSONResponse({"status":"ok"})
