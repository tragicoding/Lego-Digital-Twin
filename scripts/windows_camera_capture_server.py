"""
Windows USB camera capture relay for MINIVERSE.

Run this on Windows, not WSL:

    py -m pip install fastapi uvicorn opencv-python requests
    py scripts\\windows_camera_capture_server.py

Default camera index mapping:
    index 0 -> front
    index 1 -> left
    index 2 -> back
    index 3 -> right

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
from threading import Lock

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


@dataclass
class CameraState:
    spec: CameraSpec
    opened: bool = False
    error: str | None = None


def _camera_specs() -> list[CameraSpec]:
    return [
        CameraSpec("front", int(os.getenv("CAMERA_FRONT_INDEX", "0"))),
        CameraSpec("left", int(os.getenv("CAMERA_LEFT_INDEX", "1"))),
        CameraSpec("back", int(os.getenv("CAMERA_BACK_INDEX", "2"))),
        CameraSpec("right", int(os.getenv("CAMERA_RIGHT_INDEX", "3"))),
    ]


class CameraManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._states: dict[str, CameraState] = {
            spec.view: CameraState(spec=spec)
            for spec in _camera_specs()
        }

    def open_all(self) -> None:
        with self._lock:
            self.release_all()
            for spec in _camera_specs():
                state = CameraState(spec=spec)
                cap = cv2.VideoCapture(spec.index, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.getenv("CAMERA_WIDTH", "1280")))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.getenv("CAMERA_HEIGHT", "720")))
                if not cap.isOpened():
                    state.opened = False
                    state.error = f"camera index {spec.index} not opened"
                    cap.release()
                else:
                    state.opened = True
                    self._caps[spec.view] = cap
                self._states[spec.view] = state

            # Warm up after every camera is open. This keeps capture clicks fast.
            for _ in range(8):
                for view, cap in self._caps.items():
                    ok, _ = cap.read()
                    if not ok:
                        self._states[view].opened = False
                        self._states[view].error = "warmup frame read failed"
                time.sleep(0.03)

    def release_all(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    def health(self) -> dict:
        with self._lock:
            cameras = []
            for view in ("front", "left", "back", "right"):
                state = self._states.get(view)
                if state is None:
                    continue
                cap = self._caps.get(view)
                opened = bool(state.opened and cap is not None and cap.isOpened())
                cameras.append(
                    {
                        "view": state.spec.view,
                        "index": state.spec.index,
                        "opened": opened,
                        "error": None if opened else state.error,
                    }
                )
            ready = len(cameras) == 4 and all(camera["opened"] for camera in cameras)
            return {"status": "ok" if ready else "not_ready", "ready": ready, "cameras": cameras}

    def capture_jpeg(self, view: str) -> bytes:
        with self._lock:
            cap = self._caps.get(view)
            state = self._states.get(view)
            if cap is None or state is None or not cap.isOpened():
                raise RuntimeError(f"{view} camera is not open")

            frame = None
            for _ in range(3):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.03)

            if frame is None:
                raise RuntimeError(f"{view} camera did not return a frame")

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            if not ok:
                raise RuntimeError(f"{view} camera JPEG encode failed")
            return encoded.tobytes()


manager = CameraManager()


@app.on_event("startup")
def startup() -> None:
    manager.open_all()


@app.on_event("shutdown")
def shutdown() -> None:
    manager.release_all()


@app.get("/health")
def health():
    return manager.health()


@app.post("/reload-cameras")
def reload_cameras():
    manager.open_all()
    return manager.health()


@app.post("/capture-and-upload")
def capture_and_upload(body: CaptureRequest):
    backend = body.backend_url.rstrip("/")
    upload_url = f"{backend}/sessions/{body.session_id}/assets"

    uploaded = []
    health_payload = manager.health()
    if not health_payload["ready"]:
        raise HTTPException(status_code=503, detail=health_payload)

    for spec in _camera_specs():
        try:
            image = manager.capture_jpeg(spec.view)
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
