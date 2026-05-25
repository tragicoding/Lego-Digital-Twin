"""
realtime_motion.py
웹캠 → BlazePose → Direct IK → UDP → Unity 실시간 파이프라인

사용법:
    python realtime_motion.py [--host 127.0.0.1] [--port 7777]
    --host : Unity가 실행 중인 PC IP (WSL → Windows 는 자동 감지 시도)
    --port : BlazeMotionReceiver 에서 설정한 포트 (기본 7777)

UDP 패킷 포맷 (한 줄 텍스트):
    "root_x root_y root_z  qx0 qy0 qz0 qw0  qx1 ... qw23"
    = 3 + 4*24 = 99 개의 float, 공백 구분, \n 종료
"""

import argparse
import socket
import struct
import subprocess
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation
from PyMBS import MBS

# ── CLI 인자 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="", help="Unity UDP 수신 IP (빈 값 = 자동)")
parser.add_argument("--port", type=int, default=7777, help="UDP 포트 (기본 7777)")
parser.add_argument("--debug", action="store_true", help="viser 시각화 동시 실행")
args = parser.parse_args()

# ── WSL → Windows IP 자동 감지 ────────────────────────────────────────────────
def get_windows_host_ip():
    try:
        result = subprocess.run(
            ["cat", "/etc/resolv.conf"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("nameserver"):
                return line.split()[1]
    except Exception:
        pass
    return "127.0.0.1"

UDP_HOST = args.host if args.host else get_windows_host_ip()
UDP_PORT = args.port
print(f"[UDP] → {UDP_HOST}:{UDP_PORT}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ── MBS 스켈레톤 로드 ──────────────────────────────────────────────────────────
mbs = MBS.from_mbs_file("npc_skeleton.txt")
parents = mbs.getParents()
mbs.updateKinematicsUptoPos()

NUM_JOINTS = mbs._num_joints   # 24

# POSE_CONNECTIONS (child, parent) — root 제외
POSE_CONNECTIONS = [(i, p) for i, p in enumerate(parents) if p != -1]

# BlazePose 스켈레톤 연결 정의
BPPOSE_SKEL_LIST = [
    (11, 12),
    (12, 14), (14, 16),   # right arm
    (11, 13), (13, 15),   # left arm
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),   # left leg
    (24, 26), (26, 28),   # right leg
    (27, 31),             # left foot
    (28, 32),             # right foot
    (23, 11), (24, 12),
]

# (POSE_CONNECTIONS 인덱스, BPPOSE_SKEL_LIST 인덱스)
MAPPING_LIST = [
    (2,  8),   # L_Thigh  ↔ left hip→knee
    (3,  9),   # L_Calf   ↔ left knee→ankle
    (4,  12),  # L_Foot   ↔ left ankle→foot
    (6,  10),  # R_Thigh  ↔ right hip→knee
    (7,  11),  # R_Calf   ↔ right knee→ankle
    (8,  13),  # R_Foot   ↔ right ankle→foot
    (17, 3),   # L_Arm    ↔ left shoulder→elbow
    (18, 4),   # L_Forearm↔ left elbow→wrist
    (21, 1),   # R_Arm    ↔ right shoulder→elbow
    (22, 2),   # R_Forearm↔ right elbow→wrist
]

# ── T-포즈 기준 뼈 방향 사전 계산 ─────────────────────────────────────────────
# npc_skeleton.txt 의 local offset 을 정규화한 방향 벡터
import torch as th

def get_rest_bone_dir(pc_idx):
    """rest pose 에서의 뼈 방향 (parent local frame 기준)"""
    i_child, i_parent = POSE_CONNECTIONS[pc_idx]
    # child 의 origin translation = parent frame 에서의 오프셋
    offset = mbs._joints[i_child].origin[0, :3, 3].cpu().numpy()
    norm = np.linalg.norm(offset)
    if norm < 1e-8:
        return np.array([0.0, 1.0, 0.0])
    return offset / norm

REST_DIRS = {pc_idx: get_rest_bone_dir(pc_idx) for pc_idx, _ in MAPPING_LIST}

# ── 스무딩 버퍼 ───────────────────────────────────────────────────────────────
SMOOTH_ALPHA = 0.5   # 0=최대 스무딩, 1=스무딩 없음
prev_rotvecs = {}    # joint_idx → np.array(3,)

# ── 유틸 ──────────────────────────────────────────────────────────────────────
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def vec_to_rotvec(v_from, v_to):
    """v_from → v_to 회전을 angle-axis(rotvec)로 반환."""
    v_from = normalize(v_from)
    v_to   = normalize(v_to)
    try:
        rot, _ = Rotation.align_vectors([v_to.reshape(1, 3)],
                                         [v_from.reshape(1, 3)])
        return rot.as_rotvec()
    except Exception:
        return np.zeros(3)

def compute_pose_origin(landmarks_arr):
    """BlazePose 33개 랜드마크 배열 → 몸통 기준 좌표계 (3×4 matrix)."""
    try:
        l_sh  = landmarks_arr[11]
        r_sh  = landmarks_arr[12]
        l_hip = landmarks_arr[23]
        r_hip = landmarks_arr[24]
    except IndexError:
        return None

    shoulder_mid = (l_sh + r_sh) / 2.0
    hip_mid      = (l_hip + r_hip) / 2.0

    right_vec = normalize(l_sh - r_sh)
    body_vec  = normalize(shoulder_mid - hip_mid)
    fwd_vec   = normalize(np.cross(right_vec, body_vec))

    origin = np.eye(3, 4)
    origin[:3, 0] = np.cross(body_vec, fwd_vec)
    origin[:3, 1] = body_vec
    origin[:3, 2] = fwd_vec
    origin[:3, 3] = hip_mid
    return origin

def to_local(origin, pts):
    """pts (N×3 world) → local frame."""
    R3 = origin[:3, :3]
    t  = origin[:3, 3]
    return (pts - t) @ R3.T

def viser_to_unity(local_mat):
    """local_mat (3×4) → (pos_unity, quat_xyzw_unity)."""
    Rm = local_mat[:, :3]
    T  = local_mat[:, 3]
    pos = np.array([-T[0], T[1], T[2]])
    q   = Rotation.from_matrix(Rm).as_quat()  # x,y,z,w
    q[1] *= -1
    q[2] *= -1
    return pos, q

# ── Direct IK (LM 없음, 프레임별 직접 계산) ──────────────────────────────────
def direct_ik(bp_local):
    """
    BlazePose body-local 좌표에서 직접 joint rotvec 계산.
    bp_local: (33, 3) landmarks in body frame
    """
    global prev_rotvecs

    for pc_idx, bp_idx in MAPPING_LIST:
        i_child, i_parent = POSE_CONNECTIONS[pc_idx]

        # BlazePose 뼈 방향
        bp_from_lm, bp_to_lm = BPPOSE_SKEL_LIST[bp_idx]
        bp_dir = normalize(bp_local[bp_to_lm] - bp_local[bp_from_lm])

        # Rest pose 뼈 방향 (parent local frame)
        rest_dir = REST_DIRS[pc_idx]

        # 회전 계산
        rotvec = vec_to_rotvec(rest_dir, bp_dir)

        # 스무딩 (EMA)
        if i_parent in prev_rotvecs:
            rotvec = SMOOTH_ALPHA * rotvec + (1 - SMOOTH_ALPHA) * prev_rotvecs[i_parent]
        prev_rotvecs[i_parent] = rotvec

        # MBS joint 에 적용 (BALL joint = angle-axis 3D)
        mbs._joints[i_parent].setCompactArray(
            th.tensor(rotvec, dtype=th.float32).unsqueeze(0)
        )

    mbs.updateKinematicsUptoPos()

# ── UDP 전송 ──────────────────────────────────────────────────────────────────
def send_frame(root_origin):
    """현재 MBS 상태를 UDP로 Unity 에 전송."""
    pos_u, _ = viser_to_unity(root_origin)
    values = list(pos_u)
    for j in range(NUM_JOINTS):
        wmat = mbs._joints[j]._wmat.clone().numpy()[0]
        _, q = viser_to_unity(wmat)
        values.extend(q.tolist())

    msg = (" ".join(f"{v:.5f}" for v in values) + "\n").encode()
    sock.sendto(msg, (UDP_HOST, UDP_PORT))

# ── BlazePose 초기화 ──────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,          # 1=속도↑, 2=정확도↑
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)

print("웹캠 시작 — q 로 종료")

# ── 선택적 viser 디버그 시각화 ────────────────────────────────────────────────
viser_server = None
if args.debug:
    import viser
    viser_server = viser.ViserServer()
    mbs_pts_h = viser_server.scene.add_point_cloud(
        "/mbs_pts", points=np.zeros((NUM_JOINTS, 3)),
        colors=np.tile([[0.2, 0.9, 0.4]], (NUM_JOINTS, 1)), point_size=0.03)
    mbs_lines_h = viser_server.scene.add_line_segments(
        "/mbs_lines", points=np.zeros((len(POSE_CONNECTIONS), 2, 3)),
        colors=np.tile([[1.0, 0.8, 0.2]], (len(POSE_CONNECTIONS), 2, 1)), line_width=3.0)
    bp_pts_h = viser_server.scene.add_point_cloud(
        "/bp_pts", points=np.zeros((33, 3)),
        colors=np.tile([[0.1, 0.3, 1.0]], (33, 1)), point_size=0.025)
    bp_lines_h = viser_server.scene.add_line_segments(
        "/bp_lines", points=np.zeros((len(BPPOSE_SKEL_LIST), 2, 3)),
        colors=np.tile([[1.0, 0.0, 0.2]], (len(BPPOSE_SKEL_LIST), 2, 1)), line_width=2.0)
    print("[viser] http://localhost:4242")

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
frame_count = 0
t_prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_world_landmarks:
        # (33, 3) numpy array
        lms = np.array([[lm.x, lm.y, lm.z]
                        for lm in res.pose_world_landmarks.landmark])
        lms[:, 1] *= -1   # y 반전

        origin = compute_pose_origin(lms)
        if origin is not None:
            # body frame 변환
            bp_local = to_local(origin, lms)

            # Direct IK
            mbs._joints[0]._lmat = th.from_numpy(origin).float().unsqueeze(0)
            direct_ik(bp_local)

            # UDP 전송
            send_frame(origin)

            # viser 업데이트
            if viser_server is not None:
                mbs.getMotionMatrix()
                pts = mbs._motion_mat[:, 0, :3, 3].numpy()
                mbs_pts_h.points   = pts
                mbs_lines_h.points = np.array([[pts[c], pts[p]] for c, p in POSE_CONNECTIONS])
                bp_pts_h.points    = lms
                bp_lines_h.points  = np.array([[lms[a], lms[b]] for a, b in BPPOSE_SKEL_LIST])

        mp_drawing.draw_landmarks(
            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

    # FPS 표시
    frame_count += 1
    if frame_count % 30 == 0:
        fps = 30 / (time.time() - t_prev)
        t_prev = time.time()
        cv2.putText(frame, f"FPS:{fps:.1f}  UDP→{UDP_HOST}:{UDP_PORT}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("BlazePose Realtime", cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
print("종료")
