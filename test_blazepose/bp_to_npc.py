"""
bp_to_npc.py
BlazePose CSV → NPC FBX 스켈레톤 모션 변환
npc_skeleton.txt (24 joints) 기반 LM 최적화

Joints index:
  0=Hip(FREE), 1=Pelvis, 2=L_Thigh, 3=L_Calf, 4=L_Foot, 5=L_ToeBase,
  6=R_Thigh, 7=R_Calf, 8=R_Foot, 9=R_ToeBase,
  10=Waist, 11=Spine01, 12=Spine02, 13=NeckTwist01, 14=NeckTwist02, 15=Head,
  16=L_Clavicle, 17=L_Upperarm, 18=L_Forearm, 19=L_Hand,
  20=R_Clavicle, 21=R_Upperarm, 22=R_Forearm, 23=R_Hand
"""
import numpy as np
import torch as th
import time
import viser
import pandas as pd
from scipy.spatial.transform import Rotation as R
from lmfit import minimize, Parameters
from PyMBS import MBS

# -----------------------------------------------
# 스켈레톤 로드
# -----------------------------------------------
mbs = MBS.from_mbs_file("npc_skeleton.txt")
parents = mbs.getParents()
print("Parents:", parents)

# 기본 POSE_CONNECTIONS: 모든 non-root 본 (child, parent) 쌍
POSE_CONNECTIONS = [(i, p) for i, p in enumerate(parents) if p != -1]
# 시각화용 추가 연결: R_Hand→Waist, L_Hand→Waist
POSE_CONNECTIONS.append((23, 10))   # R_Hand to Waist
POSE_CONNECTIONS.append((19, 10))   # L_Hand to Waist

print("Num joints:", mbs._num_joints, "  DOF:", mbs._dof)
for i, (c, p) in enumerate(POSE_CONNECTIONS):
    print(f"  POSE_CONNECTIONS[{i}]: ({c},{p}) = {mbs._joints[c].name} ← {mbs._joints[p].name}")

mbs.updateKinematicsUptoPos()
default_mat = mbs.getCompactArray()
res = mbs.getCompactArray()

# -----------------------------------------------
# BlazePose 스켈레톤 연결 정의 (기존과 동일)
# BPPOSE_SKEL_LIST[idx] = (from_landmark, to_landmark)
# -----------------------------------------------
BPPOSE_SKEL_LIST = [
    (11, 12),           # 0: left-right shoulder
    (12, 14), (14, 16), # 1=right shoulder→elbow, 2=right elbow→wrist
    (11, 13), (13, 15), # 3=left shoulder→elbow,  4=left elbow→wrist
    (11, 23), (12, 24), (23, 24),  # 5,6,7: torso
    (23, 25), (25, 27), # 8=left  hip→knee,    9=left  knee→ankle
    (24, 26), (26, 28), # 10=right hip→knee,   11=right knee→ankle
    (27, 31),           # 12=left  ankle→foot
    (28, 32),           # 13=right ankle→foot
    (23, 11),           # 14=left  hip→shoulder (spine)
    (24, 12),           # 15=right hip→shoulder (spine)
]

# -----------------------------------------------
# MAPPING_LIST: (POSE_CONNECTIONS_idx, BPPOSE_SKEL_LIST_idx)
#
# POSE_CONNECTIONS[k] = (child_joint, parent_joint)
# 방향 = child_pos - parent_pos  ← 이 방향을 BlazePose 뼈 방향에 맞춤
# 제어 대상 = parent_joint 의 rotation
#
# NPC 스켈레톤:
#   POSE_CONNECTIONS[2]  = (L_Calf,   L_Thigh)   → 왼쪽 허벅지 방향
#   POSE_CONNECTIONS[3]  = (L_Foot,   L_Calf)    → 왼쪽 종아리 방향
#   POSE_CONNECTIONS[4]  = (L_ToeBase,L_Foot)    → 왼쪽 발 방향
#   POSE_CONNECTIONS[6]  = (R_Calf,   R_Thigh)   → 오른쪽 허벅지 방향
#   POSE_CONNECTIONS[7]  = (R_Foot,   R_Calf)    → 오른쪽 종아리 방향
#   POSE_CONNECTIONS[8]  = (R_ToeBase,R_Foot)    → 오른쪽 발 방향
#   POSE_CONNECTIONS[17] = (L_Forearm,L_Upperarm)→ 왼쪽 상완 방향
#   POSE_CONNECTIONS[18] = (L_Hand,   L_Forearm) → 왼쪽 전완 방향
#   POSE_CONNECTIONS[21] = (R_Forearm,R_Upperarm)→ 오른쪽 상완 방향
#   POSE_CONNECTIONS[22] = (R_Hand,   R_Forearm) → 오른쪽 전완 방향
# -----------------------------------------------
MAPPING_LIST = [
    (2,  8),  # L_Thigh direction  ↔ Blaze(23→25) left hip→knee
    (3,  9),  # L_Shin direction   ↔ Blaze(25→27) left knee→ankle
    (4,  12), # L_Foot direction   ↔ Blaze(27→31) left ankle→foot
    (6,  10), # R_Thigh direction  ↔ Blaze(24→26) right hip→knee
    (7,  11), # R_Shin direction   ↔ Blaze(26→28) right knee→ankle
    (8,  13), # R_Foot direction   ↔ Blaze(28→32) right ankle→foot
    (17, 3),  # L_Upperarm dir     ↔ Blaze(11→13) left shoulder→elbow
    (18, 4),  # L_Forearm dir      ↔ Blaze(13→15) left elbow→wrist
    (21, 1),  # R_Upperarm dir     ↔ Blaze(12→14) right shoulder→elbow
    (22, 2),  # R_Forearm dir      ↔ Blaze(14→16) right elbow→wrist
]


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def compute_pose_origin(group, up_axis='y'):
    """BlazePose 랜드마크로부터 몸통 기준 좌표계(origin) 계산."""
    try:
        l_sh  = group[group["landmark"] == 11][["x","y","z"]].values[0]
        r_sh  = group[group["landmark"] == 12][["x","y","z"]].values[0]
        l_hip = group[group["landmark"] == 23][["x","y","z"]].values[0]
        r_hip = group[group["landmark"] == 24][["x","y","z"]].values[0]
    except IndexError:
        print("랜드마크 추출 오류: frame에 필요한 landmark 없음")
        return None, None

    shoulder_mid = (l_sh + r_sh) / 2.0
    hip_mid      = (l_hip + r_hip) / 2.0

    right_vec = l_sh - r_sh
    right_vec /= np.linalg.norm(right_vec) + 1e-8

    body_vec = shoulder_mid - hip_mid
    body_vec /= np.linalg.norm(body_vec) + 1e-8

    forward_vec = np.cross(right_vec, body_vec)
    forward_vec /= np.linalg.norm(forward_vec) + 1e-8

    head_vec = np.cross(forward_vec, right_vec)
    head_vec /= np.linalg.norm(head_vec) + 1e-8

    origin = np.eye(3, 4)
    origin[:3, 2] = forward_vec
    origin[:3, 1] = body_vec
    origin[:3, 0] = np.cross(body_vec, forward_vec)
    origin[:3, 3] = hip_mid
    return forward_vec, origin


def compute_local_positions(origin, positions):
    """world 좌표 positions 를 origin 기준 local frame 으로 변환."""
    Rot = origin[:, :3]
    t   = origin[:, 3]
    pos_minus_t = positions - t.reshape(1, 3)
    return pos_minus_t @ Rot.T


# -----------------------------------------------
# LMfit 최적화 모델
# -----------------------------------------------
class OptModel_LMFIT:
    def __init__(self, nFrames, mbs):
        self.mbs = mbs
        self.nFrames = nFrames
        self.dof = len(MAPPING_LIST) * 3

    def _error_fn(self, params, tar_positions, residual):
        theta = np.array([params[f"theta_{i}"] for i in range(self.dof)])
        theta = theta.reshape(1, -1)

        residual_joint = []
        for index, (pc_idx, bp_idx) in enumerate(MAPPING_LIST):
            theta_joint = theta[:, 3*index : 3*index+3]
            _, i_parent = POSE_CONNECTIONS[pc_idx]
            residual_joint.append(residual[:, 3*i_parent : 3*i_parent+3])
            self.mbs._joints[i_parent].setCompactArray(
                th.from_numpy(theta_joint).to(th.float32)
            )
        residual_joint = np.array(residual_joint).flatten().reshape(1, -1)
        self.mbs.updateKinematicsUptoPos()

        c_test = []
        for pc_idx, bp_idx in MAPPING_LIST:
            i_child, i_parent = POSE_CONNECTIONS[pc_idx]

            # MBS 뼈 방향
            pos_child  = compute_local_positions(
                mbs._joints[0]._wmat[0],
                mbs._joints[i_child]._wmat[:, :, 3]
            )
            pos_parent = compute_local_positions(
                mbs._joints[0]._wmat[0],
                mbs._joints[i_parent]._wmat[:, :, 3]
            )
            tar_dir = normalize((pos_child - pos_parent).cpu().numpy())

            # BlazePose 뼈 방향
            bp_from, bp_to = BPPOSE_SKEL_LIST[bp_idx]
            src_dir = normalize(
                (tar_positions[bp_to, :] - tar_positions[bp_from, :]).reshape(1, -1)
            )

            c_test.append(np.abs(tar_dir - src_dir))

        c_test = np.array(c_test).flatten().reshape(1, -1)
        c_res  = np.abs(theta - residual_joint) * 0.5
        cost   = np.concatenate([c_test, c_res], axis=-1)
        return cost.flatten()

    def forward(self, tar_positions, residual):
        params = Parameters()
        for i in range(self.dof):
            params.add(f"theta_{i}", value=0.0)

        result = minimize(
            self._error_fn, params,
            args=(tar_positions, residual),
            method="leastsq", max_nfev=100
        )
        print(result.message)
        print(f"  cost: {result.chisqr:.6f}")

        theta_opt = np.array([result.params[f"theta_{i}"].value for i in range(self.dof)])
        return theta_opt[np.newaxis, :]


# -----------------------------------------------
# Viser ← Unity 좌표 변환
# -----------------------------------------------
def viser_local_to_unity(local_mat):
    """
    local_mat: shape (3,4)
    returns:
        pos_unity:  (3,)   [-X, Y, Z]
        quat_unity: (4,)   (x, y, z, w) with negated y,z
    """
    Rm = local_mat[:, :3]
    T  = local_mat[:, 3]
    px, py, pz = T
    pos_unity  = np.array([-px, py, pz])
    quat_unity = R.from_matrix(Rm).as_quat()   # x,y,z,w
    quat_unity[1] *= -1
    quat_unity[2] *= -1
    return pos_unity, quat_unity


# -----------------------------------------------
# 메인
# -----------------------------------------------
if __name__ == "__main__":
    name = "toobad"   # ← 입력 영상 이름 (pose3d_data_{name}_live.csv 필요)
    csv_path = f"pose3d_data_{name}_live.csv"

    df = pd.read_csv(csv_path)
    df["z"] = -df["z"]   # BlazePose Z 반전

    frames = df["frame"].unique()
    n_bp_points = len(df["landmark"].unique())
    start_frame = 0
    end_frames  = len(frames)
    print(f"Loaded {end_frames} frames, {n_bp_points} BlazePose landmarks per frame.")

    model = OptModel_LMFIT(nFrames=1, mbs=mbs)

    w_positions  = []
    output_txt   = []
    origin_list  = []

    for f in range(start_frame, end_frames):
        sub   = df[df["frame"] == f].sort_values("landmark")
        bppts = sub[["x", "y", "z"]].to_numpy()

        forward, origin = compute_pose_origin(sub)
        if origin is None:
            continue
        origin_list.append(origin)

        tar_positions_local = compute_local_positions(origin, bppts.copy())

        # 루트를 BlazePose 몸통 기준 좌표계에 맞춤
        mbs._joints[0]._lmat = th.from_numpy(origin).to(th.float32).unsqueeze(0)
        mbs.updateKinematicsUptoPos()

        if f == start_frame:
            residual = res.cpu().numpy()

        theta_opt = model.forward(tar_positions_local, residual)
        print(f"Frame {f}/{end_frames}")

        # 최적화 결과를 MBS 에 적용
        for index, (pc_idx, bp_idx) in enumerate(MAPPING_LIST):
            theta_joint = theta_opt[:, 3*index : 3*index+3]
            _, i_parent = POSE_CONNECTIONS[pc_idx]
            residual[:, 3*i_parent : 3*i_parent+3] = theta_opt[:, 3*index : 3*index+3]
            mbs._joints[i_parent].setCompactArray(
                th.from_numpy(theta_joint).to(th.float32)
            )

        mbs.updateKinematicsUptoPos()
        mbs.getMotionMatrix()

        # Unity 출력 포맷: root_xyz(3) + per_joint_quat(4) × num_joints
        pos_unity, _ = viser_local_to_unity(mbs._joints[0].wmat.clone().numpy()[0])
        row_values = list(pos_unity)
        for j in range(mbs._num_joints):
            _, quat = viser_local_to_unity(mbs._joints[j]._wmat.clone().numpy()[0])
            row_values.extend(quat)

        output_txt.append(" ".join(f"{v:.6f}" for v in row_values))
        w_positions.append(mbs._motion_mat[:, 0, :3, 3])

    # 결과 저장
    import os
    os.makedirs("rigmotion", exist_ok=True)
    output_path = f"rigmotion/{name}_npc_motion.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(output_txt))
    print(f"\n✅ Saved {len(output_txt)} frames → {output_path}")
    print(f"   포맷: root_xyz(3) + quat(4) × {mbs._num_joints} joints = {3 + 4*mbs._num_joints} values/frame")

    # -----------------------------------------------
    # Viser 3D 시각화
    # -----------------------------------------------
    w_positions = np.array(w_positions)

    server = viser.ViserServer()
    print("Open browser → http://localhost:4242")

    # 바닥 격자
    grid_size, grid_spacing = 2.0, 0.2
    x = np.arange(-grid_size, grid_size + grid_spacing, grid_spacing)
    z = np.arange(-grid_size, grid_size + grid_spacing, grid_spacing)
    v_lines = np.array([[[xi, 0, -grid_size], [xi, 0, grid_size]] for xi in x])
    h_lines = np.array([[[-grid_size, 0, zi], [grid_size, 0, zi]] for zi in z])
    grid_lines  = np.concatenate([v_lines, h_lines], axis=0)
    grid_colors = np.tile([[0.8, 0.8, 0.8]], (len(grid_lines), 2, 1))
    server.scene.add_line_segments("/ground_grid", points=grid_lines, colors=grid_colors, line_width=2.0)

    # BlazePose 시각화 객체
    bp_pts_handle = server.scene.add_point_cloud(
        "/pose_bp_points", points=np.zeros((n_bp_points, 3)),
        colors=np.tile([[0.1, 0.3, 1.0]], (n_bp_points, 1)), point_size=0.03)
    bp_lines_handle = server.scene.add_line_segments(
        "/pose_bp_lines",
        points=np.zeros((len(BPPOSE_SKEL_LIST), 2, 3)),
        colors=np.tile([[1.0, 0.0, 0.2]], (len(BPPOSE_SKEL_LIST), 2, 1)), line_width=3.0)

    # NPC 스켈레톤 시각화 객체
    n_joints = mbs._num_joints
    mbs_pts_handle = server.scene.add_point_cloud(
        "/pose_mbs_points", points=np.zeros((n_joints, 3)),
        colors=np.tile([[0.2, 0.9, 0.4]], (n_joints, 1)), point_size=0.035)
    mbs_lines_handle = server.scene.add_line_segments(
        "/pose_mbs_lines",
        points=np.zeros((len(POSE_CONNECTIONS), 2, 3)),
        colors=np.tile([[1.0, 0.8, 0.2]], (len(POSE_CONNECTIONS), 2, 1)), line_width=3.0)

    label_mbs = []
    for i in range(n_joints):
        label_mbs.append(server.scene.add_label(
            f"/mbs_label/{i}", text=f"{i}_{mbs._joints[i].name}",
            position=np.array([0, 0.02, 0])))

    # 루트 좌표축 프레임
    def init_frame():
        fx = server.scene.add_line_segments("/frame_x", points=np.zeros((1,2,3)),
             colors=np.array([[[1,0,0],[1,.5,.5]]]), line_width=4.0)
        fy = server.scene.add_line_segments("/frame_y", points=np.zeros((1,2,3)),
             colors=np.array([[[0,1,0],[.5,1,.5]]]), line_width=4.0)
        fz = server.scene.add_line_segments("/frame_z", points=np.zeros((1,2,3)),
             colors=np.array([[[0,0,1],[.5,.5,1]]]), line_width=4.0)
        return fx, fy, fz

    def draw_frame(fx, fy, fz, origin, scale=0.2):
        c = origin[:3, 3]
        fx.points = np.array([[c, c + origin[:3, 0] * scale]])
        fy.points = np.array([[c, c + origin[:3, 1] * scale]])
        fz.points = np.array([[c, c + origin[:3, 2] * scale]])

    fx, fy, fz = init_frame()

    try:
        while True:
            for f in range(w_positions.shape[0]):
                sub   = df[df["frame"] == f].sort_values("landmark")
                bppts = sub[["x", "y", "z"]].to_numpy()
                pts   = w_positions[f].copy()

                # BlazePose 업데이트
                bp_pts_handle.points  = bppts
                bp_lines_handle.points = np.array([[bppts[a], bppts[b]] for a, b in BPPOSE_SKEL_LIST])

                # NPC 스켈레톤 업데이트
                mbs_pts_handle.points  = pts
                mbs_lines_handle.points = np.array([[pts[c], pts[p]] for c, p in POSE_CONNECTIONS])
                for i, p in enumerate(pts):
                    label_mbs[i].position = p + np.array([0, 0.02, 0])

                # 루트 프레임
                if f < len(origin_list):
                    draw_frame(fx, fy, fz, origin_list[f])

                time.sleep(1 / 30)
    except KeyboardInterrupt:
        print("🛑 Visualization stopped.")
