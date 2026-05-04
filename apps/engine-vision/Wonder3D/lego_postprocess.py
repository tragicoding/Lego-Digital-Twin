"""
lego_postprocess.py
NeuS로 뽑은 거친 메쉬에 플라스틱 후처리를 적용한다.

처리 순서:
  1. Taubin 스무딩  - 노이즈 제거 (볼륨 수축 없음)
  2. 날카로운 엣지 보강 - 평탄면 경계를 선명하게
  3. 버텍스 컬러 보존

사용법:
  python lego_postprocess.py <input.obj> [output.obj]
  python lego_postprocess.py <input.obj> --smooth 20 --sharpen 2.0
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import trimesh.smoothing
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ── 스무딩 ────────────────────────────────────────────────────────────────────

def taubin_smooth(mesh: trimesh.Trimesh,
                  iterations: int = 15,
                  lamb: float = 0.5,
                  nu: float = -0.53) -> trimesh.Trimesh:
    """
    Taubin 스무딩: 라플라시안 스무딩 2단계를 교대로 적용.
    일반 라플라시안과 달리 메쉬가 수축하지 않음.
    """
    print(f"[SMOOTH] Taubin 스무딩 {iterations}회 (λ={lamb}, μ={nu})")
    verts = mesh.vertices.copy().astype(np.float64)
    faces = mesh.faces

    # 인접 버텍스 맵 (희소행렬)
    n = len(verts)
    adj = lil_matrix((n, n), dtype=np.float64)
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            adj[a, b] = 1
            adj[b, a] = 1
    adj = adj.tocsr()

    # 각 버텍스의 이웃 수로 정규화
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1

    for _ in range(iterations):
        # λ 스텝 (스무딩)
        neighbor_sum = adj.dot(verts)
        laplacian = neighbor_sum / deg[:, None] - verts
        verts = verts + lamb * laplacian

        # μ 스텝 (팽창 보정 → 수축 방지)
        neighbor_sum = adj.dot(verts)
        laplacian = neighbor_sum / deg[:, None] - verts
        verts = verts + nu * laplacian

    result = mesh.copy()
    result.vertices = verts
    return result


# ── 엣지 선명화 ───────────────────────────────────────────────────────────────

def sharpen_edges(mesh: trimesh.Trimesh, strength: float = 1.5) -> trimesh.Trimesh:
    """
    라플라시안 샤프닝: 스무딩의 반대 방향으로 디테일 강조.
    strength > 1: 강하게, 0~1: 약하게
    """
    print(f"[SHARP] 엣지 선명화 (strength={strength})")
    verts  = mesh.vertices.copy().astype(np.float64)
    faces  = mesh.faces
    n      = len(verts)

    adj = lil_matrix((n, n), dtype=np.float64)
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            adj[a, b] = 1
            adj[b, a] = 1
    adj = adj.tocsr()
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1

    neighbor_avg = adj.dot(verts) / deg[:, None]
    laplacian    = verts - neighbor_avg          # 원본 - 평균 = 디테일 성분
    sharpened    = verts + strength * laplacian  # 디테일 강조

    result = mesh.copy()
    result.vertices = sharpened
    return result


# ── 버텍스 컬러 보존 ──────────────────────────────────────────────────────────

def preserve_vertex_colors(original: trimesh.Trimesh,
                            processed: trimesh.Trimesh) -> trimesh.Trimesh:
    """버텍스 순서가 유지되므로 컬러를 그대로 이식"""
    if hasattr(original.visual, 'vertex_colors'):
        processed.visual.vertex_colors = original.visual.vertex_colors.copy()
    return processed


# ── 메인 ──────────────────────────────────────────────────────────────────────

def process(input_path: str, output_path: str,
            smooth_iter: int, sharpen_strength: float):

    print(f"\n[PLASTIC] 로딩: {input_path}")
    mesh = trimesh.load(input_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    print(f"[PLASTIC] 원본  버텍스: {len(mesh.vertices):,}  면: {len(mesh.faces):,}")

    # 1. Taubin 스무딩 (거친 노이즈 제거)
    smoothed = taubin_smooth(mesh, iterations=smooth_iter)

    # 2. 엣지 선명화 (플라스틱 특유의 날카로운 경계)
    sharpened = sharpen_edges(smoothed, strength=sharpen_strength)

    # 3. 버텍스 컬러 보존
    final = preserve_vertex_colors(mesh, sharpened)

    final.export(output_path)
    print(f"[PLASTIC] 저장 완료: {output_path}")
    print(f"[PLASTIC] 결과  버텍스: {len(final.vertices):,}  면: {len(final.faces):,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_obj",   help="입력 OBJ")
    p.add_argument("output_obj",  nargs="?", help="출력 OBJ (기본: input_plastic.obj)")
    p.add_argument("--smooth",    type=int,   default=15,  help="Taubin 스무딩 반복 횟수 (기본 15)")
    p.add_argument("--sharpen",   type=float, default=1.5, help="엣지 선명화 강도 (기본 1.5)")
    args = p.parse_args()

    out = args.output_obj or args.input_obj.replace(".obj", "_plastic.obj")
    process(args.input_obj, out, args.smooth, args.sharpen)
