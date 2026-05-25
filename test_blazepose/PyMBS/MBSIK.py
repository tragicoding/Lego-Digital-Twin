from PyMBS import MBS
from globalJoint import gJoint, BallJoint, FreeJoint
from VisualData import Vis

import torch
import theseus as th

mbs = MBS.from_mbs_file("mixamo_rest.txt")
print(mbs.getParents())
mbs.updateKinematicsUptoPos()

# forward kinematics 로 target position 을 생성
targetAngleAxis = torch.deg2rad(torch.tensor([[0,1 *(45),0]],dtype=torch.float32))
# target position 을 바꿈
#mbs._joints[3].setCompactArray(targetAngleAxis)
mbs._joints[1].setCompactArray(targetAngleAxis)
mbs.updateKinematicsUptoPos()
target_position = mbs._joints[9]._wmat[:,:,3].clone().detach()
base_position = mbs._joints[0]._wmat[:,:,3].clone().detach()
# theseus optimization solver 
# cost function 설계
# IK can also be solved as an optimization problem:
#      min \|log(pose^-1 * targeted_pose)\|^2
# as the following
def targeted_pose_error(optim_vars, aux_vars):
    (theta,) = optim_vars
    (targeted_pose,base_position,res) = aux_vars
    
    mbs.setFromCompactArray(theta.tensor)
    mbs.updateKinematicsUptoPos()
    
    
    #pose = th.SE3(tensor=mbs._joints[9]._wmat)
    #print(f"theta_{theta.tensor}")
    print(f"current_{mbs._joints[9]._wmat[:,:,3]}")
    print(f"target_{targeted_pose.tensor}")

    test = abs(mbs._joints[9]._wmat[:,:,3] - targeted_pose.tensor)
    test_base = abs(mbs._joints[0]._wmat[:,:,3] - base_position.tensor)
    test2 = abs(res.tensor[:,3:])
    return torch.cat([test,test_base,test2],dim=-1)


# inital 값
target_theta = torch.rand(1, mbs._dof, dtype=torch.float32)
theta_opt = torch.zeros_like(target_theta)
optim_vars = (th.Vector(tensor=(theta_opt), name="theta_opt"),)

target_pose: torch.Tensor = target_position
base_pose: torch.Tensor = base_position
res : torch.Tensor = theta_opt
aux_vars = [th.Variable(target_pose, name="targeted_position"),th.Variable(base_pose,name="base_position"),th.Variable(res,name="q")]

# LM optimize
objective = th.Objective()
cost_function = th.AutoDiffCostFunction(
    optim_vars, targeted_pose_error, 3 + 3  + 66 , aux_vars=aux_vars, cost_weight=th.ScaleCostWeight(1.0)
)

objective.add(cost_function)
layer = th.TheseusLayer(th.LevenbergMarquardt(objective, max_iterations=15,abs_err_tolerance=1e-3))

with torch.no_grad():
    teta_opt = torch.zeros_like(theta_opt)
    teta_opt[:,:3] = mbs._joints[0]._wmat[:,:,3]
    inputs = {"theta_opt": teta_opt, "targeted_pose": target_pose}
    solution, info = layer.forward(
            input_tensors=inputs,
            optimizer_kwargs={"backward_mode": "implicit"},
        )


mbs.setFromCompactArray(solution["theta_opt"])
mbs.updateKinematicsUptoPos()
print("Outer loss: ", (mbs._joints[9]._wmat[:,:,3] - target_pose).mean())
print(info.status)

# q 를 넣어서 다시 foward kinematics 로 world matrix 를 업데이트
mbs.getMotionMatrix()
vis = Vis()
world_positions = mbs._motion_mat[:,:,:,3].permute(1,0,2)
np_joints = world_positions.numpy()
np_targets = target_pose.numpy()
vis.plot_animation_pose_wTarget(np_joints,np_targets,mbs.getParents(),f"ik_out",axis_scale=100)