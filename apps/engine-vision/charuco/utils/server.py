"""
백엔드 클라이언트 모듈.

charuco 파이프라인이 복셀 JSON을 생성한 뒤 이 모듈을 통해
중앙 백엔드(apps/backend)로 POST한다.

직접 Flask 서버를 띄우지 않는다.
모든 통신은 apps/backend/main.py 를 거친다.

기본 백엔드 주소: http://localhost:8000
"""

import json
import urllib.request
import urllib.error

DEFAULT_BACKEND_URL = "http://localhost:8000"


def upload_voxels(payload: dict, backend_url: str = DEFAULT_BACKEND_URL) -> bool:
    """
    복셀 페이로드를 백엔드 서버에 업로드한다.

    Parameters
    ----------
    payload     : build_json()이 반환한 dict
    backend_url : 백엔드 서버 주소 (기본값 http://localhost:8000)

    Returns
    -------
    True  → 업로드 성공
    False → 실패 (서버 미실행 또는 네트워크 오류)
    """
    url  = f"{backend_url}/voxels/upload"
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            print(f"[backend] 업로드 완료: 복셀 {result.get('voxels')}개 "
                  f"/ Unity 클라이언트 {result.get('unity_clients')}개 전송")
            return True
    except urllib.error.URLError as e:
        print(f"[backend] 업로드 실패 — 백엔드 서버가 실행 중인지 확인하세요. ({e.reason})")
        return False
