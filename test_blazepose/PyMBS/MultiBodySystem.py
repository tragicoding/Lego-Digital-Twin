import abc
import torch
from torchlie.functional.constants import DeviceType
from typing import Dict, List, Optional

import numpy as np
from .globalJoint import gJoint, BallJoint, FreeJoint, PrismaticJoint, RevoluteJoint, FixedJoint
import math
from torchlie.functional import SE3

        
class MBSJointInfo:
    def __init__(self, name, parent=None, origin_xyz=None, origin_rpy=None, joint_type=None, axis=None):
        self.name = name
        self.parent = parent
        self.origin_xyz = origin_xyz
        self.origin_rpy = origin_rpy
        self.joint_type = joint_type
        self.axis = axis
        
class MBS(abc.ABC):
    def __init__(self, name: str, dtype: torch.dtype = None, device: DeviceType = None):
        self._name: str = name
        self._dtype: torch.dtype = (
            dtype if dtype is not None else torch.get_default_dtype()
        )
        self._device: torch.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self._dof: int = 0
        self._num_joints: int = 0
        self._joints: List[gJoint] = []
        self._joint_map: Dict[str, gJoint] = {}
        self._motion_mat: None
        self._q : torch.Tensor = torch.zeros(1,self._dof, dtype=dtype, device=device)

    def setDefaultPos(self):
        for joint in self._joints:
            joint._lmat = joint.origin
    
    def getJointMotionMatrix(self,id):
        return self._joints[id].wmat

    def getMotionMatrix(self):
        self._motion_mat = self._joints[0].wmat.new_empty(self._num_joints, self._joints[0].wmat.shape[0],3,4)
        for i in range(0,self._num_joints):
            self._motion_mat[i] = self.getJointMotionMatrix(i)

    def getParents(self):
        parent_indices =[]
        for name, joint in self._joint_map.items():
            
            if joint.parent is None :
                parent_idx = -1
            else:
                parent_name = joint.parent.name
                parent_idx = list(self._joint_map.keys()).index(parent_name)
            parent_indices.append(parent_idx)
        parent_indices_np = np.array(parent_indices)
        return parent_indices_np

    def transFromCompactArray(self,q):
        if(q.shape[-1] != self._dof):
            print("dof is not equal to vector shape ")
        
        joint_id = 0
        for j, joint in enumerate(self._joints):
            if(j!=0):
                joint.relative_pose(q[:,joint_id:joint_id+3])
                joint_id = joint_id + joint.dof
            else:
                joint.translation(q[:,:3])
                joint.relative_pose(q[:,3:6])
                joint_id = joint_id + joint.dof

    def setFromCompactArray(self,q):
        if(q.shape[-1] != self._dof):
            print("dof is not equal to vector shape ")
        
        joint_id = 0
        for j, joint in enumerate(self._joints):
            if(j!=0):
                joint.setCompactArray(q[:,joint_id:joint_id+3])
                joint_id = joint_id + joint.dof
            else:
                joint.setCompactArray(q[:,:6])
                joint_id = joint_id + joint.dof

    def getCompactArray(self):
        joint_id = 0
        q_c = self._joints[0].getCompactArray()
        
        q = torch.zeros(q_c.shape[0],self._dof, dtype=self._dtype, device=self._device)
        for j, joint in enumerate(self._joints):
            q_c = joint.getCompactArray()
            q[:,joint_id:joint_id+joint.dof] = q_c
            joint_id = joint_id + joint.dof
        
        return q
    
    def updateKinematicsUptoPos(self):
        self._joints[0].update_kinematicpose()

    @staticmethod
    def from_mbs_file(
        mbs_file: str, dtype: torch.dtype = None, device: DeviceType = None
    ) -> "MBS":
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")

        # .txt 파일에서 정보를 읽어옴
        def read_hierarchy_info_from_file(file_path):
            with open(file_path, 'r') as file:
                return file.read()

        # HIERARCHY 정보를 바탕으로 MBSJoint 객체 생성
        def parse_hierarchy_info(hierarchy_info):
            joints = []
            current_joint_info = {}
            cnt = 0
            lines = hierarchy_info.strip().split('\n')
            for i , line in enumerate(lines):
                line = line.strip()
                if line.startswith("LINK"):
                    current_joint_info = {}
                elif line.startswith("NAME"):
                    current_joint_info['name'] = line.split(" ")[1]
                elif line.startswith("PARENT"):
                    current_joint_info['parent'] = line.split(" ")[1]
                elif line.startswith("POS"):
                    current_joint_info['origin_xyz'] = [float(x) for x in line.strip().split()[1:] if x.strip() != '']
                elif line.startswith("ROT"):
                    quat = [float(x) for x in line.strip().split()[2:] if x.strip() != '']
                    rpy = quaternion_to_euler_angle(quat[3],quat[0],quat[1],quat[2])
                    current_joint_info['origin_rpy'] = rpy
                elif line.startswith("JOINT"):
                    current_joint_info['joint_type'] = line.split(" ")[2]
                elif line.startswith("END_LINK"):
                    if current_joint_info:
                        joint = MBSJointInfo(**current_joint_info)
                        joints.append(joint)
                    pass
                elif line.startswith("END_HIERARCHY"):
                    break
            
            return joints
        
        def quaternion_to_euler_angle(w, x, y, z):
            """
            쿼터니언을 롤, 피치, 요로 변환하는 함수
            :param w: 쿼터니언의 w 성분
            :param x: 쿼터니언의 x 성분
            :param y: 쿼터니언의 y 성분
            :param z: 쿼터니언의 z 성분
            :return: 롤, 피치, 요 각도 (radians)
            """
            # 쿼터니언을 회전 행렬로 변환
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            sinp = 2.0 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)  # 피치가 90도일 때
            else:
                pitch = math.asin(sinp)

            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            return [roll, pitch, yaw]


        #urdf_model = urdf.URDF.from_xml_file(urdf_file)
        info = read_hierarchy_info_from_file(mbs_file)
        mbs_joints_info = parse_hierarchy_info(info)
        mbs = MBS('MBS', dtype, device)

       
        
        
    
        def get_joint_type(jointtype: str):
            if jointtype == "FREE":
                return FreeJoint
            elif jointtype == "BALL":
                return BallJoint
            else:
                raise ValueError(f"{jointtype} is currently not supported.")

        def get_origin(urdf_origin_xyz = None, urdf_origin_rpy =None) -> torch.Tensor:
            origin = torch.eye(3, 4, dtype=dtype, device=device).unsqueeze(0)
            if urdf_origin_xyz is None:
                return origin

            if urdf_origin_xyz is not None:
                origin[:, :, 3] = origin.new_tensor(urdf_origin_xyz)

            if urdf_origin_rpy is not None:
                rpy = origin.new_tensor(urdf_origin_rpy)
                c3, c2, c1 = rpy.cos()
                s3, s2, s1 = rpy.sin()
                origin[:, 0, 0] = c1 * c2
                origin[:, 0, 1] = (c1 * s2 * s3) - (c3 * s1)
                origin[:, 0, 2] = (s1 * s3) + (c1 * c3 * s2)
                origin[:, 1, 0] = c2 * s1
                origin[:, 1, 1] = (c1 * c3) + (s1 * s2 * s3)
                origin[:, 1, 2] = (c3 * s1 * s2) - (c1 * s3)
                origin[:, 2, 0] = -s2
                origin[:, 2, 1] = c2 * s3
                origin[:, 2, 2] = c2 * c3

            return origin

       
        def create_joints():
            for i, mbs_joint in enumerate(mbs_joints_info):
                # find joint type
                joint_type = get_joint_type(mbs_joint.joint_type)
                # origin (default matrix)
                origin = get_origin(mbs_joint.origin_xyz,mbs_joint.origin_rpy)
            
                # create joints following hierarchy
                joint = joint_type(
                        mbs_joint.name,
                        origin=origin,
                        id= i
                    )
                # save (name, gjoint) list      
                mbs._joint_map[mbs_joint.name] = joint

        def build_kinematicsChain(mbs,mbs_joints_info):
            mbs._joints = []
            for i, mbs_joint in enumerate(mbs_joints_info):
                name = mbs_joint.name
                
                child_list=[]
                for mbs_joint_cand in mbs_joints_info:
                    if(mbs_joint_cand.parent == name):
                        child_list.append(mbs._joint_map[mbs_joint_cand.name])

                # add kinematic chain
                mbs._joint_map[name]._child = child_list

                if(i !=0):
                    parentname = mbs_joint.parent
                    mbs._joint_map[name].setParent(mbs._joint_map[parentname])

                # get final joint info
                mbs._joints.append(mbs._joint_map[name])

                mbs._dof += mbs._joints[i].dof

        create_joints()
        build_kinematicsChain(mbs,mbs_joints_info)

        mbs._num_joints = len(mbs._joints)

        return mbs

    @staticmethod
    def from_urdf_file(
        urdf_file: str, dtype: torch.dtype = None, device: DeviceType = None
    ) -> "MBS":
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")

        urdf_model = urdf.URDF.from_xml_file(urdf_file)
        robot = MBS(urdf_model.name, dtype, device)

        def get_joint_type(joint: urdf.Joint):
            if joint.type == "revolute" or joint.type == "continuous":
                return RevoluteJoint
            elif joint.type == "prismatic":
                return PrismaticJoint
            elif joint.type == "fixed":
                return FixedJoint
            else:
                raise ValueError(f"{joint.type} is currently not supported.")

        def get_origin(urdf_origin: Optional[urdf.Pose] = None) -> torch.Tensor:
            origin = torch.eye(3, 4, dtype=dtype, device=device).unsqueeze(0)
            if urdf_origin is None:
                return origin

            if urdf_origin.xyz is not None:
                origin[:, :, 3] = origin.new_tensor(urdf_origin.xyz)

            if urdf_origin.rpy is not None:
                rpy = origin.new_tensor(urdf_origin.rpy)
                c3, c2, c1 = rpy.cos()
                s3, s2, s1 = rpy.sin()
                origin[:, 0, 0] = c1 * c2
                origin[:, 0, 1] = (c1 * s2 * s3) - (c3 * s1)
                origin[:, 0, 2] = (s1 * s3) + (c1 * c3 * s2)
                origin[:, 1, 0] = c2 * s1
                origin[:, 1, 1] = (c1 * c3) + (s1 * s2 * s3)
                origin[:, 1, 2] = (c3 * s1 * s2) - (c1 * s3)
                origin[:, 2, 0] = -s2
                origin[:, 2, 1] = c2 * s3
                origin[:, 2, 2] = c2 * c3

            return origin

        def create_joints(robot):

            for i, urdf_joint in enumerate(urdf_model.joints):
                # origin (default matrix)
                origin = get_origin(urdf_joint.origin)
                # find joint type
                joint_type = get_joint_type(urdf_joint)

                # joint
                if joint_type is FixedJoint:
                    joint = joint_type(
                        urdf_joint.name,
                        origin=origin,
                        id = i
                    )
                else:
                    axis = origin.new_tensor(urdf_joint.axis)
                    joint = joint_type(
                        urdf_joint.name,
                        axis,
                        origin=origin,
                        id = i 
                    )
                robot._joint_map[urdf_joint.name] = joint
        
        def build_kinematicsChain(robot):
            robot._joints = []
            for m, k in robot._joint_map:
                print(m)
            for i, mbs_joint in enumerate(robot._joint_map):
                name = mbs_joint.name
                
                child_list=[]
                for mbs_joint_cand in robot._joint_map:
                    if(mbs_joint_cand.parent == name):
                        child_list.append(robot._joint_map[mbs_joint_cand.name])

                # add kinematic chain
                robot._joint_map[name]._child = child_list

                if(i !=0):
                    parentname = mbs_joint.parent
                    robot._joint_map[name].setParent(robot._joint_map[parentname])

                # get final joint info
                robot._joints.append(robot._joint_map[name])

                robot._dof += robot._joints[i].dof

        # move the child joints of any fixed joint to its parent link
        def simplify_kinematics_tree(mbs):
            for _, link in mbs._joint_map.items():
                for joint in link._child:
                    if isinstance(joint, FixedJoint):
                        subjoints: List[gJoint] = joint._child
                        # removing children of fixed joint to avoid traversing twice
                        joint._child_joints = []
                        for subjoint in subjoints:
                            subjoint._parent = link
                            subjoint._origin = SE3.compose(
                                joint.origin, subjoint.origin
                            )
                            link._child.append(subjoint)

        
       
        create_joints(robot)
        build_kinematicsChain(robot)                
        simplify_kinematics_tree(robot)

        robot._num_joints = len(robot.joints)

        return robot

        
if __name__ == "__main__":  
    mbs = MBS.from_mbs_file("mixamo_rest.txt")
    print(mbs.getParents())
    print(mbs._name)
    print(mbs._num_joints)
    print(mbs._dof)

    tensor_list = [[0,0,0]] * 60
    tensor_list = torch.tensor(tensor_list,dtype=torch.float32)

    angles = torch.linspace(0, 90, 60)
    rotation_Axis = []
    for angle in angles:
        angleAxis = torch.tensor([0, 1, 0],dtype=torch.float32)
        angle_radian = torch.deg2rad(angle)
        angleAxis = angleAxis * angle_radian
        rotation_Axis.append(angleAxis)
    rot_Axis = torch.stack(rotation_Axis)

    for j in range(0, mbs._num_joints):
        if(j == 3):
            #mbs._joints[j].relative_pose(angle = rot_Axis)
            mbs._joints[j].setCompactArray(rot_Axis)
        else:
            mbs._joints[j].relative_pose(angle = tensor_list)

#    target_theta = torch.rand(1, mbs._dof, dtype=torch.float32)
#    mbs.setFromCompactArray(target_theta)
    mbs.getCompactArray()

    mbs.updateKinematicsUptoPos()
    for i in range(mbs._num_joints):
        print(mbs._joints[i].name)
        print(mbs._joints[i].wmat)
    mbs.getMotionMatrix()
    vis = Vis()

    world_positions = mbs._motion_mat[:,:,:,3].permute(1,0,2)
    np_joints = world_positions.numpy()
    vis.plot_animation_pose(np_joints, mbs.getParents(),filename=f"test_{0}", axis_scale=100)