import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uuid, os, asyncio

app = FastAPI()

# Check GPU/CPU
if torch.cuda.is_available():
    device = "cuda"
    print("🚀 Using GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("⚡ Using CPU fallback")

UPLOAD_DIR = "/tmp/uploads"
OUTPUT_DIR = "/tmp/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": f"Motion server running on {device.upper()}!"}

@app.post("/animate")
async def animate(
    file: UploadFile = File(...),
    cue: str = Form(None)  # text cue for camera/body movement
):
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Load image
    img = cv2.imread(input_path)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Convert to tensor + push to GPU
    tensor = torch.from_numpy(img).to(device)
    tensor = tensor.float() / 255.0
    img = (tensor * 255).byte().to("cpu").numpy()

    h, w, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

    # Animation frames
    for i in range(300):  # ~10s at 30fps
        frame = img.copy()

        # Apply "cues"
        if cue:
            if "zoom" in cue.lower():
                scale = 1 + (i / 300) * 0.5  # gradual zoom
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, 0, scale)
                frame = cv2.warpAffine(frame, M, (w, h))

            if "pan left" in cue.lower():
                M = np.float32([[1, 0, -i/10], [0, 1, 0]])
                frame = cv2.warpAffine(frame, M, (w, h))

            if "pan right" in cue.lower():
                M = np.float32([[1, 0, i/10], [0, 1, 0]])
                frame = cv2.warpAffine(frame, M, (w, h))

            if "tilt" in cue.lower():
                M = cv2.getRotationMatrix2D((w//2, h//2), i/30, 1.0)
                frame = cv2.warpAffine(frame, M, (w, h))

        out.write(frame)

    out.release()

    # Cleanup task
    asyncio.create_task(cleanup_file(output_path, delay=5))
    return FileResponse(output_path, media_type="video/mp4")

async def cleanup_file(path, delay=5):
    await asyncio.sleep(delay)
    try:
        os.remove(path)
    except:
        pass
