"""
Tripo3D API 클라이언트
image_to_model → (character only) rig → animate → download


Tripo API를 호출하는 함수들이 모여있는 파일.

upload_image()
이미지 파일 업로드 → image_token 반환

create_model_task()
image_token으로 3D 모델 생성 task 등록 → task_id 반환

create_rig_task()
3D 모델 task_id로 리깅 task 등록 → task_id 반환

create_animation_task()
리깅 task_id로 애니메이션 task 등록 → task_id 반환

poll_task()
task_id 상태를 반복 조회 → 완료되면 결과 반환

download_glb()
결과에서 GLB URL 추출 → 서버에 .glb 파일 저장

"""

'''
asyncio: 비동기 대기 처리용
httpx: Python에서 HTTP 요청을 보내는 라이브러리
Path: 파일 경로를 객체처럼 다루기 위한 클래스
httpx.AsyncClient()를 쓰기 때문에 전체 함수들이 async def'''
import asyncio
import httpx
from pathlib import Path

#API 설정값
from ..core.config import TRIPO_API_KEY, TRIPO_BASE_URL
#인증 헤더
_HEADERS = {"Authorization": f"Bearer {TRIPO_API_KEY}"}

#이미지 파일을 Tipo에 업로드 → file_token 반환
async def upload_image(image_path: Path) -> str:
    """이미지를 업로드하고 file_token을 반환한다."""
    ext = image_path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"
    async with httpx.AsyncClient() as c:
        with open(image_path, "rb") as f: #rb는 read binary
            resp = await c.post(
                f"{TRIPO_BASE_URL}/upload/sts",
                headers=_HEADERS,
                files={"file": (image_path.name, f, mime)},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.json()["data"]["image_token"]

#이미지를 3D 모델로 변환하는 task를 생성하는 함수.
async def create_model_task(file_token: str) -> str:
    """image_to_model 태스크를 생성하고 task_id를 반환한다."""
    async with httpx.AsyncClient() as c:
        resp = await c.post(
            f"{TRIPO_BASE_URL}/task",
            headers={**_HEADERS, "Content-Type": "application/json"},
            json={"type": "image_to_model", "file": {"type": "png", "file_token": file_token}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"]["task_id"]
    #이 task_id는 나중에 poll_task()로 작업 완료 여부를 확인할 때 사용

#이미 생성된 3D 모델에 리깅 작업을 요청하는 함수.
async def create_rig_task(model_task_id: str) -> str:
    """리깅 태스크를 생성하고 task_id를 반환한다."""
    async with httpx.AsyncClient() as c:
        resp = await c.post(
            f"{TRIPO_BASE_URL}/task",
            headers={**_HEADERS, "Content-Type": "application/json"},
            json={"type": "animate_rig", "original_model_task_id": model_task_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"]["task_id"]

#리깅된 모델에 애니메이션을 적용하는 task를 생성하는 함수.
async def create_animation_task(rig_task_id: str, animation: str) -> str:
    """
    애니메이션 태스크 생성.
    animation: "preset:walk" | "preset:wave" 등 Tripo 지원 preset
    """
    async with httpx.AsyncClient() as c:
        resp = await c.post(
            f"{TRIPO_BASE_URL}/task",
            headers={**_HEADERS, "Content-Type": "application/json"},
            json={
                "type": "animate_retarget",
                "original_model_task_id": rig_task_id,
                "animation": animation,
                "out_format": "glb",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"]["task_id"]

#Tripo task가 끝났는지 계속 확인하는 함수. 
#interval = 3초마다 확인
#timout = 최대 300초까지 대기. 
async def poll_task(task_id: str, interval: float = 3.0, timeout: float = 300.0) -> dict:
    """태스크 완료까지 polling하고 result dict를 반환한다."""
    elapsed = 0.0
    async with httpx.AsyncClient() as c:
        while elapsed < timeout:
            resp = await c.get(
                f"{TRIPO_BASE_URL}/task/{task_id}",
                headers=_HEADERS,
                timeout=30,
            ) #현재 task의 현재 상태를 Tripo API에서 조회
            resp.raise_for_status()
            data = resp.json()["data"]
            status = data["status"] #status는 "processing", "success", "failed" 등 작업 상태를 나타냄
            #현재 task 상태를 꺼낸다.

            if status in ("success", "FINISHED"):
                return data
            if status in ("failed", "FAILED", "cancelled", "unknown"):
                raise RuntimeError(f"Tripo 태스크 실패: {status} (task_id={task_id})")

            await asyncio.sleep(interval)
            elapsed += interval

    raise TimeoutError(f"Tripo 태스크 타임아웃 ({timeout}s): {task_id}")
'''
task_id 확인
→ 아직 처리 중이면 3초 대기
→ 다시 확인
→ 성공하면 결과 반환
→ 실패하면 예외 발생
→ 너무 오래 걸리면 TimeoutError'''


#Tripo 결과에서 GLB 파일 URL을 찾아서 다운로드 하는 함수. 
async def download_glb(result: dict, output_path: Path) -> Path:
    """GLB 파일을 다운로드한다."""
    glb_url = result["output"]["pbr_model"]
    async with httpx.AsyncClient() as c:
        resp = await c.get(glb_url, timeout=120, follow_redirects=True)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
    return output_path

"""
character_task.py와 연결되는 흐름

앞에서 본 캐릭터 파이프라인에서는 이 함수들이 이렇게 사용된다.

token = await tripo.upload_image(Path(asset.input_image_path))

이미지 업로드.

model_task_id = await tripo.create_model_task(token)
model_result = await tripo.poll_task(model_task_id)

3D 모델 생성.

rig_task_id = await tripo.create_rig_task(model_task_id)
rig_result = await tripo.poll_task(rig_task_id)

리깅.

anim_task_id = await tripo.create_animation_task(rig_task_id, anim["preset"])
anim_result = await tripo.poll_task(anim_task_id)
await tripo.download_glb(anim_result, glb_path)

애니메이션 생성 후 GLB 다운로드.
"""
