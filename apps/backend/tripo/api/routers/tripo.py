"""
Tripo 라우터

엔드포인트:
    GET  /tripo/status/{task_id}  — 변환 태스크 상태 조회
    POST /tripo/upload            — 수동 이미지 업로드 → 3D 변환
"""

import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ...api.routers import _task_store as tasks
from ...schemas.tripo import TaskStatus

router = APIRouter(prefix="/tripo", tags=["tripo"])


@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "task_id를 찾을 수 없습니다.")
    data = tasks[task_id]
    return TaskStatus(task_id=task_id, **data)


@router.post("/upload")
async def manual_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """수동으로 이미지를 업로드해 3D 변환을 시작한다 (디버깅/테스트용)."""
    from ...api.routers.hardware import _run_tripo_pipeline

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 가능합니다.")

    task_id = str(uuid.uuid4())
    image_bytes = await file.read()
    tasks[task_id] = {"status": "queued", "filename": file.filename}
    background_tasks.add_task(_run_tripo_pipeline, task_id, image_bytes, file.filename)
    return {"task_id": task_id, "status": "queued"}
