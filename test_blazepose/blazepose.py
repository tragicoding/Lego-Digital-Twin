import cv2, mediapipe as mp, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
import time
import viser

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
print("✅ 완료: output_pose.mp4 & pose3d_data.csv 저장됨")


## post processing 
csv_path = "pose3d_data_toobad_live.csv"  # BlazePose에서 저장한 파일 경로
df = pd.read_csv(csv_path)
df["y"] = -df["y"]  # y축 반전
adjusted = []
for f, group in df.groupby("frame"):
    # 두 발 (31: left_foot_index, 32: right_foot_index)
    try:
        z_left = group[group["landmark"] == 31]["y"].values[0]
        z_right = group[group["landmark"] == 32]["y"].values[0]
    except IndexError:
        continue  # foot landmark 없음 → skip
   
    # 가장 낮은 발이 바닥
    ground_z = min(z_left, z_right)

    # 전체 포즈 z좌표 보정
    group = group.copy()
    group["y"] = group["y"] - ground_z  # 바닥을 z=0으로 정규화

    adjusted.append(group)

df_adj = pd.concat(adjusted)
df_adj.to_csv(csv_path, index=False)
df = pd.read_csv(csv_path)

# frame 단위로 그룹화
frames = df["frame"].unique()
landmarks = df["landmark"].unique()
n_points = len(landmarks)
print(f"Loaded {len(frames)} frames, {n_points} landmarks per frame.")

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

            time.sleep(1 / 30)  # 30 FPS

except KeyboardInterrupt:
    print("🛑 Visualization stopped.")