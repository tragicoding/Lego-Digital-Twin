import numpy as np
import torch as th
import time
import viser
import pandas as pd
from scipy.spatial.transform import Rotation as R
from lmfit import minimize, Parameters
from PyMBS import MBS

mbs = MBS.from_mbs_file("RetargetedYbot.txt")
parents = mbs.getParents()
print(parents)
POSE_CONNECTIONS = [(i, p) for i, p in enumerate(parents) if p != -1]
POSE_CONNECTIONS.append((21,6)) # right lh->sh
POSE_CONNECTIONS.append((14,1)) # left lh->sh


mbs.updateKinematicsUptoPos() # 모든 joint에다가 4x4 matrix update


default_mat = mbs.getCompactArray() # base position 3개 + 25개 local rotation get data 
res = mbs.getCompactArray()
def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v  # 혹은 np.zeros_like(v)
    return v / norm

for i,joint in enumerate(POSE_CONNECTIONS):
    print(f"index: {i}, list: ({joint[0]},{joint[1]}) : {mbs._joints[joint[0]].name} , {mbs._joints[joint[1]].name}")



MAPPING_LIST = [
    (1, 8),   # Left thigh  : Mix(1,0) ↔ Blaze(23,25)
    (2, 9),   # Left shin   : Mix(2,1) ↔ Blaze(25,27)
    (3, 12),  # left foot  : Mix(7,6) ↔ Blaze(26,28)
    (6, 10),  # Right thigh : Mix(6,0) ↔ Blaze(24,26)
    (7, 11),  # Right shin  : Mix(7,6) ↔ Blaze(26,28)
    (8, 13),  # Right foot : Mix(6,0) ↔ Blaze(24,26)
    (15, 3),  # Left upper arm : Mix(14,13) ↔ Blaze(11,12)
    (16, 4),  # Left lower arm : Mix(15,14) ↔ Blaze(12,14)
    (22, 1),  # Right upper arm: Mix(21,13) ↔ Blaze(11,13)
    (23, 2),  # Right lower arm: Mix(22,21) ↔ Blaze(13,15)
    # (24, 15), # Right thigh -> upper arm
    # (25, 14), # Left  thigh -> upper arm 
]
# BlazePose 기본 Pose 연결 정의 (대략적인)
BPPOSE_SKEL_LIST = [
    (11, 12), (12, 14), (14, 16), (11, 13), (13, 15),  # arms
    (11, 23), (12, 24), (23, 24),                     # torso
    (23, 25), (25, 27), (24, 26), (26, 28),           # 8-9 right legs 10-11 left legs
    (27, 31), (28, 32),                            # 12 right ankle 13 left ankle 
    (23, 11), (24, 12)                             # 14 right lh->sh 15 left lh->sh
]
BPPOSE_SKEL_LIST = [
    (11, 12), 
    (12, 14), (14, 16), # right arms (1,2)
    (11, 13), (13, 15),  # left arms (3,4)
    (11, 23), (12, 24), (23, 24),                     # torso
    (23, 25), (25, 27), # left legs (8,9)
    (24, 26), (26, 28), # right legs (10,11)          
    (27, 31), # left ankle (12)
    (28, 32), # right angkle (13)
    (23, 11), # left thigh->shoulder (14)
    (24, 12)  # right thigh->shoulder (15)
]


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
    right_vec = l_sh - r_sh
    right_vec /= np.linalg.norm(right_vec) + 1e-8

    # 몸통 축 (어깨-엉덩이 중심선)
    body_vec = shoulder_mid - hip_mid
    body_vec /= np.linalg.norm(body_vec) + 1e-8

    # forward 벡터 = up, right, body 로부터 cross product로 정의
    # 어깨가 x축, 위가 y축이므로 forward는 (right x up)
    forward_vec = np.cross(right_vec, body_vec)
    forward_vec /= np.linalg.norm(forward_vec) + 1e-8

    # body head
    head_vec = np.cross(forward_vec,right_vec)
    head_vec /= np.linalg.norm(head_vec) + 1e-8
    # 몸의 실제 기울기 보정 (어깨-엉덩이 방향과 일관성 확인)
    # 만약 forward와 body_vec이 반대 방향이면 뒤집음
    #if np.dot(forward_vec, body_vec) < 1e-5:
    #    forward_vec = -forward_vec
    origin = np.eye(3, 4)
    origin[:3,2] = forward_vec
    origin[:3,1] = body_vec
    origin[:3,0] = np.cross(body_vec,forward_vec)
    origin[:3,3] = hip_mid
    return forward_vec, origin

def compute_local_positions(origin,positions):
    # origin: shape (3,4)
    R = origin[:, :3]       # (3,3)
    t = origin[:, 3]        # (3,)

    # positions: shape (33,3)
    pos_minus_t = positions - t.reshape(1,3)     # (33,3)

    positions_local = pos_minus_t @ R.T          # (33,3)
    return positions_local

# --------------------------------------------------
# LMfit-based Optimization Model
# --------------------------------------------------
class OptModel_LMFIT:
    def __init__(self, nFrames, mbs):
        self.mbs = mbs
        self.nFrames = nFrames
        self.dof = mbs._dof
        self.POSE_CONNECTIONS = [(i, p) for i, p in enumerate(parents) if p != -1]
        self.MAPPING_LIST = []

    # error function (residuals)
    def _error_fn(self, params, tar_positions, residual):
        theta = np.array([params[f"theta_{i}"] for i in range(self.dof)])
        theta = theta.reshape(1, -1)
        
        residual_joint = []
        for index, (mix_idx, bp_idx) in enumerate(MAPPING_LIST):
            theta_joint = theta[:,3*index:3*index+3]
            
            _, i_parent = POSE_CONNECTIONS[mix_idx]
            residual_joint.append(residual[:,3*i_parent:3*i_parent+3])

            mbs._joints[i_parent].setCompactArray(th.from_numpy(theta_joint).to(th.float32))
        residual_joint = np.array(residual_joint).flatten().reshape(1,-1)
        self.mbs.updateKinematicsUptoPos()
        
        c_test = []
        for mix_idx, bp_idx in MAPPING_LIST:
            
            # --- Mixamo bone ---
            i_child, i_parent =  POSE_CONNECTIONS[mix_idx]
            pos_local_child = compute_local_positions(mbs._joints[0]._wmat[0],
                                                      mbs._joints[i_child]._wmat[:,:,3]
                                                      )
            pos_local_parent = compute_local_positions(mbs._joints[0]._wmat[0],
                                                      mbs._joints[i_parent]._wmat[:,:,3]
                                                      )
            
            tar_dir = pos_local_child - pos_local_parent
            tar_dir = normalize(tar_dir.cpu().numpy())

            # --- BlazePose bone ---
            i_parent, i_child = BPPOSE_SKEL_LIST[bp_idx]
            src_dir = tar_positions[i_child,:] - tar_positions[i_parent,:]
            src_dir = normalize(src_dir.reshape(1,-1))

            # --- loss ---
            loss = tar_dir - src_dir

            # print("loss", loss)
            c_test.append(np.abs(loss))
        c_test = np.array(c_test).flatten().reshape(1,-1)

        # simplified cost similar to Theseus version
        c_res = np.abs(theta-residual_joint) * 0.5

        # flatten()
        cost = np.concatenate([c_test, c_res], axis=-1)
        return cost.flatten()

    def forward(self, tar_positions, residual):
        # define parameters
        params = Parameters()
        for i in range(self.dof):
            params.add(f"theta_{i}", value=0.0)

        # run LM optimization
        result = minimize(self._error_fn, params,
                          args=(tar_positions, residual),
                          method="leastsq",max_nfev=100)  # Levenberg-Marquardt
        print(result.message)
        print(f"Final cost: {result.chisqr:.6f}")

        # extract optimized parameters
        theta_opt = np.array([result.params[f"theta_{i}"].value for i in range(self.dof)])
        theta_opt = theta_opt[np.newaxis,:]
        return theta_opt
        

# --------------------------------------------------
# Example Usage
# --------------------------------------------------
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
    pos_unity = np.array([-px, py, pz])   # flip Z

    # --- Rotation 변환 ---
    # S = np.diag([1, 1, -1])
    # R_unity = S @ Rm @ S

    # --- Quaternion 추출 ---
    quat_unity = R.from_matrix(Rm).as_quat()  # (x,y,z,w)
    quat_unity[1] = -1 * quat_unity[1]
    quat_unity[2] = -1 * quat_unity[2]
    return pos_unity, quat_unity

if __name__ == "__main__":
    # input data
    name = '1999'
    csv_path = f"pose3d_data_{name}_live.csv"  # BlazePose에서 저장한 파일 경로
    df = pd.read_csv(csv_path)
    df["z"] = -df["z"]  # y축 반전
    # frame 단위로 그룹화
    frames = df["frame"].unique()
    landmarks = df["landmark"].unique()
    n_bp_points = len(landmarks)

    ### CHECK!!! YOUR SEQUENTIAL DATA ###
    end_frames = len(frames)
    start_frame = 0
    print(f"Loaded {len(frames)} frames, {n_bp_points} landmarks per frame.")

    # mbs (multibodysystem)
    model = OptModel_LMFIT(nFrames=1, mbs=mbs)
    model.dof = len(MAPPING_LIST)*3

    w_positions =[]
    output_txt = []
    origin_list = []
    for f in range(start_frame,end_frames): #frames:
        sub = df[df["frame"] == f].sort_values("landmark")
        bppts = sub[["x", "y", "z"]].to_numpy()
        
        # compute origin
        forward,origin = compute_pose_origin(sub)
        origin_list.append(origin)
        # 
        tar_positions_local = compute_local_positions(origin, bppts.copy())
        
        mbs._joints[0]._lmat = th.from_numpy(origin).to(th.float32).unsqueeze(0)
        mbs.updateKinematicsUptoPos()
    
        # optimization
        tar_positions = tar_positions_local
        if f == start_frame :
            residual = res.cpu().numpy()
        theta_opt = model.forward(tar_positions, residual)
        print(f"{f} / {end_frames} ") #: Optimized θ: {theta_opt} ")

        for index, (mix_idx, bp_idx) in enumerate(MAPPING_LIST):
            theta_joint = theta_opt[:,3*index:3*index+3]
            
            _, i_parent = POSE_CONNECTIONS[mix_idx]
            residual[:,3*i_parent:3*i_parent+3] = theta_opt[:,3*index:3*index +3]

            mbs._joints[i_parent].setCompactArray(th.from_numpy(theta_joint).to(th.float32))

        # save output pose
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
    output_path = f"rigmotion/{name}_motion.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(output_txt))

    print(f"Saved frames → {output_path}")
        
    w_positions = np.array(w_positions)



    
    # -----------------------------------------------
    # 2️⃣ Viser 서버 시작
    # -----------------------------------------------
    server = viser.ViserServer()
    print("Open browser → http://localhost:4242")

    
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
    
    init_bp_pts = np.zeros((n_bp_points, 3))
    init_bp_colors = np.tile(np.array([[0.1, 0.3, 1.0]]), (n_bp_points, 1))

    bppose_points = server.scene.add_point_cloud(
        name="/pose_bp_points",
        points=init_bp_pts,
        colors=init_bp_colors,
        point_size=0.03,
    )

    label_nodes = []
    for i, p in enumerate(init_bp_pts):
        node = server.scene.add_label(
            name=f"/pose_bp_labels/landmark_{i}",
            text=str(i),
            position=p + np.array([0,0.02,0]),
        )
        label_nodes.append(node)

    seg_bp_points = np.zeros((len(BPPOSE_SKEL_LIST), 2, 3))
    seg_bp_colors = np.tile(np.array([[1.0, 0.0, 0.2]]), (len(BPPOSE_SKEL_LIST), 2, 1))

    bppose_lines = server.scene.add_line_segments(
        name="/pose_bp_lines",
        points=seg_bp_points,
        colors=seg_bp_colors,
        line_width=3.0,
    )
    

    n_points = 25
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

    label_mbs_nodes = []
    for i, p in enumerate(init_pts):
        node_mbs = server.scene.add_label(
            name=f"/pose_mbs_labels/landmark_{i}",
            text=str(f"{i}_{mbs._joints[i].name}"),
            position=p + np.array([0,0.02,0]),
        )
        label_mbs_nodes.append(node_mbs)

    
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

            for f in range(0,w_positions.shape[0]): #frames:
                
                sub = df[df["frame"] == f].sort_values("landmark")
                bppts = sub[["x", "y", "z"]].to_numpy()
                
                pts = w_positions[f].copy()
                
                # ✅ 객체 핸들을 통해 직접 업데이트
                bppose_points.points = bppts
                seg_bp_points = np.array([[bppts[a], bppts[b]] for (a, b) in BPPOSE_SKEL_LIST])
                bppose_lines.points = seg_bp_points
                for i, p in enumerate(bppts):
                    label_nodes[i].position = p + np.array([0, 0.02, 0])


                # ✅ 객체 핸들을 통해 직접 업데이트
                pose_points.points = pts
                seg_points = np.array([[pts[a], pts[b]] for (a, b) in POSE_CONNECTIONS])
                pose_lines.points = seg_points
                for i, p in enumerate(pts):
                    label_mbs_nodes[i].position = p + np.array([0, 0.02, 0])

                # Root Frame Estimation
                origin = origin_list[f]
                draw_frame(frame_x,frame_y,frame_z,origin)
                
                time.sleep(1 / 30)  # 30 FPS

               

    except KeyboardInterrupt:
        print("🛑 Visualization stopped.")