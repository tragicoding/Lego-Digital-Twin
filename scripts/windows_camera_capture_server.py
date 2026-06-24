"""
Windows USB camera capture relay for MINIVERSE.

Run this on Windows, not WSL:

    py -m pip install fastapi uvicorn opencv-python requests
    py scripts\\windows_camera_capture_server.py

Default camera indices:
    front=0, left=1, back=2, right=3

Override with environment variables if Windows assigns different indices:
    set CAMERA_FRONT_INDEX=0
    set CAMERA_LEFT_INDEX=1
    set CAMERA_BACK_INDEX=2
    set CAMERA_RIGHT_INDEX=3

The WSL backend calls:
    POST http://<Windows-IP>:3100/capture-and-upload
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="MINIVERSE Windows Camera Capture")


class CaptureRequest(BaseModel):
    backend_url: str
    session_id: str
    asset_type: str = "object"


@dataclass(frozen=True)
class CameraSpec:
    view: str
    index: int


def _camera_specs() -> list[CameraSpec]:
    return [
        CameraSpec("front", int(os.getenv("CAMERA_FRONT_INDEX", "0"))),
        CameraSpec("left", int(os.getenv("CAMERA_LEFT_INDEX", "1"))),
        CameraSpec("back", int(os.getenv("CAMERA_BACK_INDEX", "2"))),
        CameraSpec("right", int(os.getenv("CAMERA_RIGHT_INDEX", "3"))),
    ]


def _capture_jpeg(spec: CameraSpec) -> bytes:
    cap = cv2.VideoCapture(spec.index, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"camera {spec.index} not opened")

        # Warm up a few frames so exposure/white balance settles.
        frame = None
        for _ in range(5):
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)

        if frame is None:
            raise RuntimeError(f"camera {spec.index} did not return a frame")

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        if not ok:
            raise RuntimeError(f"camera {spec.index} JPEG encode failed")
        return encoded.tobytes()
    finally:
        cap.release()


@app.get("/health")
def health():
    return {"status": "ok", "cameras": [spec.__dict__ for spec in _camera_specs()]}


@app.post("/capture-and-upload")
def capture_and_upload(body: CaptureRequest):
    backend = body.backend_url.rstrip("/")
    upload_url = f"{backend}/sessions/{body.session_id}/assets"

    uploaded = []
    for spec in _camera_specs():
        try:
            image = _capture_jpeg(spec)
            response = requests.post(
                upload_url,
                data={"asset_type": body.asset_type, "view": spec.view},
                files={"file": (f"{body.asset_type}_{spec.view}.jpg", image, "image/jpeg")},
                timeout=120,
            )
            response.raise_for_status()
            uploaded.append({"view": spec.view, "camera_index": spec.index, "response": response.json()})
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{spec.view} camera/upload failed: {exc}",
            ) from exc

    return {"status": "ok", "session_id": body.session_id, "uploaded": uploaded}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CAMERA_SERVER_PORT", "3100")))
