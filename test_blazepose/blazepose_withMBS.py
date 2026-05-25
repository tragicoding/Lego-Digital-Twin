import cv2, mediapipe as mp, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
import torch 
import time
import viser
from scipy.spatial.transform import Rotation as R

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 🔹 선택: 0 = 웹캠 / 파일 = 동영상
use_webcam = False
name = "toobad"
video_path = f"videos/{name}.mp4"

if use_webcam:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter(
    "output_pose_sodapop_live.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

pose_rows = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_world_landmarks:
        for i, lm in enumerate(res.pose_world_landmarks.landmark):
            pose_rows.append({
                "frame": frame_idx,
                "landmark": i,
                "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility
            })
        mp_drawing.draw_landmarks(
            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )
        
    cv2.imshow("BlazePose 3D", cv2.flip(frame, 1))
    out.write(cv2.flip(frame, 1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

pd.DataFrame(pose_rows).to_csv(f"pose3d_data_{name}_live.csv", index=False)

## post processing 
def compute_pose_origin(group, up_axis='y'):
    
    try:
        # 주요 랜드마크 추출
        l_sh = group[group["landmark"] == 11][["x","y","z"]].values[0]
        r_sh = group[group["landmark"] == 12][["x","y","z"]].values[0]
        l_hip = group[group["landmark"] == 23][["x","y","z"]].values[0]
        r_hip = group[group["landmark"] == 24][["x","y","z"]].values[0]
    except IndexError:
        print("wrong indexing: there is no landmark in here")

    # 어깨 중앙과 엉덩이 중앙
    shoulder_mid = (l_sh + r_sh) / 2.0
    hip_mid = (l_hip + r_hip) / 2.0

    # 위쪽 방향 (up vector)
    if up_axis == 'y':
        up = np.array([0, 1, 0])
    elif up_axis == 'z':
        up = np.array([0, 0, 1])
    else:
        raise ValueError("up_axis must be 'y' or 'z'")

    # 좌우 벡터 (어깨 방향)
    right_vec = r_sh - l_sh
    right_vec /= np.linalg.norm(right_vec) + 1e-8

    # 몸통 축 (어깨-엉덩이 중심선)
    body_vec = shoulder_mid - hip_mid
    body_vec /= np.linalg.norm(body_vec) + 1e-8

    # forward 벡터 = up, right, body 로부터 cross product로 정의
    # 어깨가 x축, 위가 y축이므로 forward는 (right x up)
    forward_vec = np.cross(right_vec, up)
    forward_vec /= np.linalg.norm(forward_vec) + 1e-8

    # 몸의 실제 기울기 보정 (어깨-엉덩이 방향과 일관성 확인)
    # 만약 forward와 body_vec이 반대 방향이면 뒤집음
    #if np.dot(forward_vec, body_vec) < 1e-5:
    #    forward_vec = -forward_vec
    origin = np.eye(3, 4)
    origin[:3,2] = forward_vec
    origin[:3,1] = up
    origin[:3,0] = np.cross(up,forward_vec)
    origin[:3,3] = hip_mid
    return forward_vec, origin

csv_path = f"pose3d_data_{name}_live.csv"  # BlazePose에서 저장한 파일 경로
df = pd.read_csv(csv_path)
df["y"] = -df["y"]  # y축 반전
# df["z"] = -df["z"]  # z축 반전

adjusted = []
forward_list = []
origin_list = []
for f, group in df.groupby("frame"):
    # 두 발 (31: left_foot_index, 32: right_foot_index)
    try:
        z_left = group[group["landmark"] == 31]["y"].values[0]
        z_right = group[group["landmark"] == 32]["y"].values[0]
        # 주요 랜드마크 추출
        l_sh = group[group["landmark"] == 11][["x","y","z"]].values[0]
        r_sh = group[group["landmark"] == 12][["x","y","z"]].values[0]
        l_hip = group[group["landmark"] == 23][["x","y","z"]].values[0]
        r_hip = group[group["landmark"] == 24][["x","y","z"]].values[0]
    except IndexError:
        continue  # foot landmark 없음 → skip
    
    # 가장 낮은 발이 바닥
    ground_z = min(z_left, z_right)

    # 전체 포즈 z좌표 보정
    group = group.copy()
    group["y"] = group["y"] - ground_z  # 바닥을 z=0으로 정규화

    # compute origin
    forward,origin = compute_pose_origin(group)
    forward_list.append(forward)
    origin_list.append(origin)
    
    adjusted.append(group)

df_adj = pd.concat(adjusted)
df_adj.to_csv(csv_path, index=False)
print("✅ 완료: output_pose.mp4 & pose3d_data.csv 저장됨")
df = pd.read_csv(csv_path)

# frame 단위로 그룹화
frames = df["frame"].unique()
landmarks = df["landmark"].unique()
n_points = len(landmarks)
print(f"Loaded {len(frames)} frames, {n_points} landmarks per frame.")

from PyMBS import MBS

mbs = MBS.from_mbs_file("RetargetedYbot.txt")
parents = mbs.getParents()
print(parents)
MBS_POSE_CONNECTIONS = [(i, p) for i, p in enumerate(parents) if p != -1]

mbs.updateKinematicsUptoPos() # 모든 joint에다가 4x4 matrix update

def viser_local_to_unity(local_mat):
    """
    local_mat: shape (3,4)
    returns:
        pos_unity: (3,)
        quat_unity: (4,) (x,y,z,w)
    """

    Rm = local_mat[:, :3]   # rotation 3x3
    T = local_mat[:, 3]     # translation

    # --- Position 변환 ---
    px, py, pz = T
    pos_unity = np.array([px, py, -pz])   # flip Z

    # --- Rotation 변환 ---
    S = np.diag([1, 1, -1])
    R_unity = S @ Rm @ S

    # --- Quaternion 추출 ---
    quat_unity = R.from_matrix(R_unity).as_quat()  # (x,y,z,w)

    return pos_unity, quat_unity

w_positions =[]
output_txt = []
for f in frames:
    sub = df[df["frame"] == f].sort_values("landmark")
    bppts = sub[["x", "y", "z"]].to_numpy()
    
    # optimization
    
    # save output pose
    pelvis_mat = torch.from_numpy(origin_list[f]).unsqueeze(0).to(torch.float32)
    mbs._joints[0]._lmat = pelvis_mat
    mbs.updateKinematicsUptoPos()
    
    motion_mat = mbs.getMotionMatrix()
    pos_unity, _ = viser_local_to_unity(mbs._joints[0].wmat.clone().numpy()[0])
    row_values = list(pos_unity)  # position 3개 먼저 추가
    for j in range(0,mbs._num_joints):
        _,quat = viser_local_to_unity(mbs._joints[j]._wmat.clone().numpy()[0])
        row_values.extend(quat)

    output_txt.append(" ".join(f"{v:.6f}" for v in row_values))
    w_positions.append(mbs._motion_mat[:,0,:3,3])

# 파일 저장
output_path = "test.txt"
with open(output_path, "w") as f:
    f.write("\n".join(output_txt))

print(f"Saved frames → {output_path}")
    
w_positions = np.array(w_positions)

# -----------------------------------------------
# 2️⃣ Viser 서버 시작
# -----------------------------------------------
server = viser.ViserServer()
print("Open browser → http://localhost:4242")

# BlazePose 기본 Pose 연결 정의 (대략적인)
POSE_CONNECTIONS = [
    (11, 12), (12, 14), (14, 16), (11, 13), (13, 15),  # arms
    (11, 23), (12, 24), (23, 24),                     # torso
    (23, 25), (25, 27), (24, 26), (26, 28),           # legs
    (27, 29), (29, 31), (28, 30), (30, 32)            # lower legs
]

grid_size = 2.0      # 전체 크기 (단위: 미터)
grid_spacing = 0.2   # 셀 간격 (격자 크기)
line_thickness = 2.0 # 선 두께

# 격자선 좌표 만들기
x = np.arange(-grid_size, grid_size + grid_spacing, grid_spacing)
z = np.arange(-grid_size, grid_size + grid_spacing, grid_spacing)

# 세로선 (x 고정, z 변)
vertical_lines = np.array([
    [[xi, 0, -grid_size], [xi, 0, grid_size]] for xi in x
])

# 가로선 (z 고정, x 변)
horizontal_lines = np.array([
    [[-grid_size, 0, zi], [grid_size, 0, zi]] for zi in z
])

# 모든 선 합치기
grid_lines = np.concatenate([vertical_lines, horizontal_lines], axis=0)

# 색상 (교차 격자 느낌을 위해 밝은 회색)
grid_colors = np.tile(np.array([[0.8, 0.8, 0.8]]), (len(grid_lines), 2, 1))

# Viser에 추가
server.scene.add_line_segments(
    name="/ground_grid",
    points=grid_lines,
    colors=grid_colors,
    line_width=line_thickness,
)

# -----------------------------------------------
# 3️⃣ 초기 시각화 객체 생성
# -----------------------------------------------
init_pts = np.zeros((n_points, 3))
init_colors = np.tile(np.array([[0.2, 0.7, 1.0]]), (n_points, 1))

pose_points = server.scene.add_point_cloud(
    name="/pose_points",
    points=init_pts,
    colors=init_colors,
    point_size=0.03,
)

seg_points = np.zeros((len(POSE_CONNECTIONS), 2, 3))
seg_colors = np.tile(np.array([[1.0, 0.8, 0.2]]), (len(POSE_CONNECTIONS), 2, 1))

pose_lines = server.scene.add_line_segments(
    name="/pose_lines",
    points=seg_points,
    colors=seg_colors,
    line_width=3.0,
)

forward_line = server.scene.add_line_segments(
    name="/forward_vector",
    points=np.zeros((1, 2, 3)),
    colors=np.array([[[1.0, 0.0, 0.0], [1.0, 0.5, 0.5]]]),  # 빨간색 계열
    line_width=5.0,
)


n_points = 25
init_pts = np.zeros((n_points, 3))
init_colors = np.tile(np.array([[0.9, 0.7, 1.0]]), (n_points, 1))

mbs_joint_points = server.scene.add_point_cloud(
    name="/mbs_joints",
    points=init_pts,
    colors=init_colors,
    point_size=0.03,
)

seg_points = np.zeros((len(MBS_POSE_CONNECTIONS), 2, 3))
seg_colors = np.tile(np.array([[1.0, 0.0, 0.2]]), (len(MBS_POSE_CONNECTIONS), 2, 1))

mbs_skeleton_lines = server.scene.add_line_segments(
    name="/mbs_skels",
    points=seg_points,
    colors=seg_colors,
    line_width=3.0,
)

def init_frame():
    frame_x = server.scene.add_line_segments(
    name="/frame_x",
    points=np.zeros((1, 2, 3)),  # (start, end)
    colors=np.array([[[1.0, 0.0, 0.0], [1.0, 0.5, 0.5]]]),  # red
    line_width=4.0,
    )

    frame_y = server.scene.add_line_segments(
        name="/frame_y",
        points=np.zeros((1, 2, 3)),
        colors=np.array([[[0.0, 1.0, 0.0], [0.5, 1.0, 0.5]]]),  # green
        line_width=4.0,
    )

    frame_z = server.scene.add_line_segments(
        name="/frame_z",
        points=np.zeros((1, 2, 3)),
        colors=np.array([[[0.0, 0.0, 1.0], [0.5, 0.5, 1.0]]]),  # blue
        line_width=4.0,
    )
    return frame_x, frame_y, frame_z

def draw_frame(frame_x, frame_y, frame_z, origin, scale=0.3):
    """
    Viser에서 주어진 origin에 좌표축 frame(XYZ)을 그림.
    origin: (x,y,z) 또는 numpy array
    scale: 각 축 길이
    """
 
    origin = np.array(origin, dtype=float)
    x_axis = origin[:3,0]
    y_axis = origin[:3,1]
    z_axis = origin[:3,2]
    center = origin[:3,3]
    
    frame_x.points = np.array([[center, center + x_axis * scale]])
    frame_y.points = np.array([[center, center + y_axis * scale]])
    frame_z.points = np.array([[center, center + z_axis * scale]])

frame_x,frame_y,frame_z = init_frame()

# -----------------------------------------------
# 4️⃣ 애니메이션 루프 (핸들 기반 업데이트)
# -----------------------------------------------
try:
    while True:

        for f in frames:
            sub = df[df["frame"] == f].sort_values("landmark")
            pts = sub[["x", "y", "z"]].to_numpy()
            #pts[:, 2] *= -1  # Z축 반전 (시각화용)

            # ✅ 객체 핸들을 통해 직접 업데이트
            pose_points.points = pts
            seg_points = np.array([[pts[a], pts[b]] for (a, b) in POSE_CONNECTIONS])
            pose_lines.points = seg_points
            
            #  몸 중심 (어깨, 엉덩이 평균)
            l_sh = pts[11]; r_sh = pts[12]
            l_hip = pts[23]; r_hip = pts[24]
            center = (l_sh + r_sh + l_hip + r_hip) / 4.0

            # Forward 벡터 가져오기 (normalize됨)
            fwd = forward_list[f]
            scale = 0.5  # forward 화살표 길이 (조정 가능)
            start = center
            end = center + fwd * scale
            forward_line.points = np.array([[start, end]])

            # Root Frame Estimation
            origin = origin_list[f]
            draw_frame(frame_x,frame_y,frame_z,origin)

            # Draw MBS 
            mbs_pts = w_positions[f]
            mbs_joint_points.points = mbs_pts
            seg_points = np.array([[mbs_pts[a], mbs_pts[b]] for (a, b) in MBS_POSE_CONNECTIONS])
            mbs_skeleton_lines.points = seg_points

            time.sleep(1 / 30)  # 30 FPS

except KeyboardInterrupt:
    print("🛑 Visualization stopped.")