import abc
import warnings

import torch
from torchlie.functional import SE3
from torchlie.functional.constants import DeviceType
from typing import List, Optional
from torchlie.functional import SO3
import copy

class gJoint(abc.ABC):
    def __init__(
        self,
        name: str,
        dof: int,
        id: Optional[int] = None,
        parent_jnt: Optional['gJoint'] = None,
        child_jnt: Optional[List['gJoint']] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        if origin is None and dtype is None:
            dtype = torch.get_default_dtype()
        if origin is None and device is None:
            device = torch.device("cpu")
        if origin is not None:
            self._origin = origin

            if dtype is not None and origin.dtype != dtype:
                warnings.warn(
                    f"The origin's dtype {origin.dtype} does not match given dtype {dtype}, "
                    "Origin's dtype will take precendence."
                )
            dtype = origin.dtype

            if device is not None and origin.device != device:
                warnings.warn(
                    f"tensor.device {origin.device} does not match given device {device}, "
                    "tensor.device will take precendence."
                )
            dtype = origin.dtype
            device = origin.device
        else:
            origin = torch.zeros(1, 3, 4, dtype=dtype, device=device)
            origin[:, 0, 0] = 1
            origin[:, 1, 1] = 1
            origin[:, 2, 2] = 1

        self._name = name
        self._dof = dof
        self._id = id
        self._parent = parent_jnt
        self._child = child_jnt
        self._origin = origin # default matrix
        self._wmat = origin # world matrix
        self._lmat = origin # local matrix
        self._dtype = dtype
        self._device = torch.device(device)
        self._axis: torch.Tensor = torch.zeros(6, self.dof, dtype=dtype, device=device)
        self._q : torch.Tensor = torch.zeros(1,self.dof, dtype=dtype, device=device)

    def setParent(self, gJ_P:'gJoint'):
        self._parent = gJ_P
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def id(self) -> int:
        return self._id

    @property
    def parent(self) -> Optional['gJoint']:
        return self._parent

    @property
    def child(self) -> Optional[List['gJoint']]:
        return self._child

    @property
    def origin(self) -> torch.Tensor:
        return self._origin

    @property
    def lmat(self) -> torch.Tensor:
        return self._lmat

    @property
    def wmat(self) -> torch.Tensor:
        return self._wmat

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def setMotion(self,nFrame):
        self._wmat = self._wmat.repeat(nFrame, 1, 1)
        self._lmat = self._lmat.repeat(nFrame, 1, 1)
        self._origin = self._origin.repeat(nFrame, 1, 1)
    
    @property
    def axis(self) -> torch.Tensor:
        return self._axis

    @abc.abstractmethod
    def getCompactArray(self):
        pass

    @abc.abstractmethod
    def setCompactArray(self, exp):
        pass

    @abc.abstractmethod
    def relative_pose(self, angle: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def update_kinematicpose(self):
        pass

#--- Robot joint revolute joint
class _RevoluteJointImpl(gJoint):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: gJoint = None,
        child_joint: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name=name, dof=1, id=id, parent_jnt=parent_joint, 
                         child_jnt=child_joint, origin=origin, dtype=dtype, device=device)

    @abc.abstractmethod
    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _check_input(angle: torch.Tensor):
        if angle.ndim == 1:
            angle = angle.view(-1, 1)
        if angle.ndim != 2 or angle.shape[1] != 1:
            raise ValueError("The joint angle must be a vector.")
        return angle
    
    def rotation(self, angle: torch.Tensor) -> torch.Tensor:
        return self._rotation_impl(angle)
    
    def _relative_pose_impl(self, angle: torch.Tensor) -> torch.Tensor:
        rot = self.rotation(angle)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3] @ rot
        ret[:, :, 3] = self.origin[:, :, 3]
        return ret

    def relative_pose(self, angle: Optional[torch.Tensor] = None) -> torch.Tensor:
        #print("update local rotation")
        angle = _RevoluteJointImpl._check_input(angle)
        self._lmat = self._relative_pose_impl(angle)
        return self._lmat
    
    def update_kinematicpose(self):
        #print("update world origin matrix")
        if(self.parent == None):
            self.parent._wmat = self.lmat.new_empty(self.lmat.shape[0],3,4)
            self.parent.wmat[:, 0, 0] = 1
            self.parent.wmat[:, 1, 1] = 1
            self.parent.wmat[:, 2, 2] = 1
        else:
            self._wmat = SE3.compose(self.parent.wmat, self.lmat)

        if(self.child is not None):
            for child in self.child:
                child.update_kinematicpose()
    
    def getCompactArray(self):
        self._q = SE3.log(self.lmat)[:,3:]
        return self._q
    
    def _setCompactArray_impl(self,angle):
        return self.relative_pose(angle)
    
    def setCompactArray(self, angle):
        angle = _RevoluteJointImpl._check_input(angle)
        self._lmat =  self._setCompactArray_impl(angle)

class RevoluteJoint(_RevoluteJointImpl):
    def __init__(
        self,
        name: str,
        revolute_axis: torch.Tensor,
        id: Optional[int] = None,
        parent_joint: Optional[gJoint] = None,
        child_joints: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        if revolute_axis.numel() != 3:
            raise ValueError("The revolute axis must have 3 elements.")

        super().__init__(name, id, parent_joint, child_joints, origin, dtype, device)

        if revolute_axis.dtype != self.dtype:
            raise ValueError(f"The dtype of revolute_axis should be {self.dtype}.")

        if revolute_axis.device != self.device:
            raise ValueError(f"The device of revolute_axis should be {self.device}.")

        self._axis[3:] = revolute_axis.view(-1, 1)

    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        return SO3.exp(angle @ self.axis[3:].view(1, -1))
    
class _RevoluteJointXYZImpl(_RevoluteJointImpl):
    axis_info = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]

    def __init__(
        self,
        name: str,
        axis_id: int,
        id: Optional[int] = None,
        parent_joint: Optional[gJoint] = None,
        child_joints: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, parent_joint, child_joints, origin, dtype, device)
        if axis_id < 0 or axis_id >= 3:
            raise ValueError("The axis_id must be one of (0, 1, 2).")
        self._axis_id = axis_id
        self._axis[self.axis_id + 3] = 1

    @property
    def axis_id(self) -> int:
        return self._axis_id

    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        info = _RevoluteJointXYZImpl.axis_info[self.axis_id]
        rot = angle.new_zeros(angle.shape[0], 3, 3)
        rot[:, info[0], info[0]] = 1
        rot[:, info[1], info[1]] = angle.cos()
        rot[:, info[2], info[2]] = rot[:, info[1], info[1]]
        rot[:, info[2], info[1]] = angle.sin()
        rot[:, info[1], info[2]] = -rot[:, info[2], info[1]]

        return rot

class RevoluteJointX(_RevoluteJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: Optional[gJoint] = None,
        child_joints: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 0, id, parent_joint, child_joints, origin, dtype, device)

class RevoluteJointY(_RevoluteJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: Optional[gJoint] = None,
        child_joints: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 1, id, parent_joint, child_joints, origin, dtype, device)

class RevoluteJointZ(_RevoluteJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: Optional[gJoint] = None,
        child_joints: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 2, id, parent_joint, child_joints, origin, dtype, device)

#--- Robot joint prismatic joint
class _PrismaticJointImpl(gJoint):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: gJoint = None,
        child_joint: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 1, id, parent_joint, child_joint, origin, dtype, device)

    @abc.abstractmethod
    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _check_input(angle: torch.Tensor):
        if angle.ndim == 1:
            angle = angle.view(-1, 1)
        if angle.ndim != 2 or angle.shape[1] != 1:
            raise ValueError("The joint angle must be a vector.")
        return angle

    def translation(self, angle: torch.Tensor) -> torch.Tensor:
        angle = _PrismaticJointImpl._check_input(angle)
        return self._translation_impl(angle)

    def _relative_pose_impl(self, angle: torch.Tensor) -> torch.Tensor:
        # "The locally origin position of joint, as seen from parent joint, moves by amount equal to the angle position from initial position"
        trans = self.translation(angle)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3]
        ret[:, :, 3:] = (
            self.origin[:, :, :3] @ trans.view(-1, 3, 1) + self.origin[:, :, 3:]
        )
        return ret
     
    def relative_pose(self, trans) -> torch.Tensor:
        trans = _PrismaticJointImpl._check_input(trans)
        self._lmat = self._relative_pose_impl(trans)
        return self._lmat

    def update_kinematicpose(self):
        #print("update world origin matrix")
        if(self.parent == None):
            self.parent._wmat = self.lmat.new_empty(self.lmat.shape[0],3,4)
            self.parent.wmat[:, 0, 0] = 1
            self.parent.wmat[:, 1, 1] = 1
            self.parent.wmat[:, 2, 2] = 1
        else:
            self._wmat = SE3.compose(self.parent.wmat, self.lmat)

        if(self.child is not None):
            for child in self.child:
                child.update_kinematicpose()
    
    def getCompactArray(self):
        self._q = (self.lmat)[:,:,3]
        return self._q
    
    def _setCompactArray_impl(self,angle):
        return self.relative_pose(angle)
    
    def setCompactArray(self, angle):
        angle = _PrismaticJointImpl._check_input(angle)
        self._lmat =  self._setCompactArray_impl(angle)

class PrismaticJoint(_PrismaticJointImpl):
    def __init__(
        self,
        name: str,
        prismatic_axis: torch.Tensor,
        id: Optional[int] = None,
        parent_joint: gJoint = None,
        child_joint: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        if prismatic_axis.numel() != 3:
            raise ValueError("The prismatic axis must have 3 elements.")

        super().__init__(name, id, parent_joint, child_joint, origin, dtype, device)

        if prismatic_axis.dtype != self.dtype:
            raise ValueError(f"The dtype of prismatic_axis should be {self.dtype}.")

        if prismatic_axis.device != self.device:
            raise ValueError(f"The device of prismatic_axis should be {self.device}.")

        self._axis[:3] = prismatic_axis.view(-1, 1)

    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        return angle @ self.axis[:3].view(1, -1)

class _PrismaticJointXYZImpl(_PrismaticJointImpl):
    def __init__(
        self,
        name: str,
        axis_id: int,
        id: Optional[int] = None,
        parent_joint: gJoint = None,
        child_joint: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, parent_joint, child_joint, origin, dtype, device)
        if axis_id < 0 or axis_id >= 3:
            raise ValueError("The axis_id must be one of (0, 1, 2).")
        self._axis_id = axis_id
        self._axis[axis_id] = 1

    @property
    def axis_id(self) -> int:
        return self._axis_id

    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        trans = angle.new_zeros(angle.shape[0], 3)
        trans[:, self.axis_id] = angle
        return trans

    def _relative_pose_impl(self, angle: torch.Tensor) -> torch.Tensor:
        _PrismaticJointXYZImpl._check_input(angle)
        angle = angle.view(-1, 1)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3]
        ret[:, :, 3] = angle * self.origin[:, :, self.axis_id] + self.origin[:, :, 3]
        return ret

class PrismaticJointX(_PrismaticJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: gJoint = None,
        child_joint: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 0, id, parent_joint, child_joint, origin, dtype, device)

class PrismaticJointY(_PrismaticJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: gJoint = None,
        child_joint: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 1, id, parent_joint, child_joint, origin, dtype, device)

class PrismaticJointZ(_PrismaticJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: gJoint = None,
        child_joint: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 2, id, parent_joint, child_joint, origin, dtype, device)

#--- Robot joint fixed joint
class FixedJoint(gJoint):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: Optional[gJoint] = None,
        child_joints: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 0, id, parent_joint, child_joints, origin, dtype, device)

    def relative_pose(self, angle) -> torch.Tensor:
        return self.origin

    def update_kinematicpose(self):
        #print("update world origin matrix")
        if(self.parent == None):
            self.parent._wmat = self.lmat.new_empty(self.lmat.shape[0],3,4)
            self.parent.wmat[:, 0, 0] = 1
            self.parent.wmat[:, 1, 1] = 1
            self.parent.wmat[:, 2, 2] = 1
        else:
            self._wmat = SE3.compose(self.parent.wmat, self.lmat)

        if(self.child is not None):
            for child in self.child:
                child.update_kinematicpose()
    
    def getCompactArray(self):
        self._q = (self.lmat)[:,:,3]
        return self._q
    
    def _setCompactArray_impl(self,angle):
        return self.relative_pose(angle)
    
    def setCompactArray(self, angle):
        self._lmat =  self._setCompactArray_impl(angle)

#--- human joint
class BallJoint(gJoint):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: gJoint = None,
        child_joint: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name=name, dof=3, id=id, parent_jnt=parent_joint, 
                         child_jnt=child_joint, origin=origin, dtype=dtype, device=device)

    @staticmethod
    def _check_input(angle: torch.Tensor):
        if angle.ndim == 1:
            angle = angle.view(-1, 3)
        if angle.ndim != 2:
            raise ValueError("The joint angle must be a exponential coordinate vector.")
        return angle
    
    def _rotation_impl(self, exp: torch.Tensor) -> torch.Tensor:
        return SO3.exp(exp)

    def rotation(self, angle: torch.Tensor) -> torch.Tensor:
        return self._rotation_impl(angle)
 
    def _relative_pose_impl(self, angle: torch.Tensor) -> torch.Tensor:
        rot = self.rotation(angle)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3] @ rot
        ret[:, :, 3] = self.origin[:, :, 3]
        return ret
    
    def relative_pose(self, angle: Optional[torch.Tensor] = None) -> torch.Tensor:
        #print("update local rotation")
        angle = BallJoint._check_input(angle)
        self._lmat = self._relative_pose_impl(angle)
        return self._lmat
    
    def update_kinematicpose(self):
        #print("update world origin matrix")
        if(self.parent == None):
            self.parent._wmat = self.lmat.new_empty(self.lmat.shape[0],3,4)
            self.parent.wmat[:, 0, 0] = 1
            self.parent.wmat[:, 1, 1] = 1
            self.parent.wmat[:, 2, 2] = 1
        else:
            self._wmat = SE3.compose(self.parent.wmat, self.lmat)

        if(self.child is not None):
            for child in self.child:
                child.update_kinematicpose()
    
    def getCompactArray(self):
        self._q = SE3.log(self.lmat)[:,3:]
        return self._q
    
    def _setCompactArray_impl(self,exp):
        ret = exp.new_empty(exp.shape[0], 3, 4)
        ret[:, :, :3] = SO3.exp(exp)
        ret[:,:,3] = self._origin[:,:,3]        
        return ret
    
    def setCompactArray(self, exp):
        exp = BallJoint._check_input(exp)
        self._lmat =  self._setCompactArray_impl(exp)

class FreeJoint(gJoint):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: gJoint = None,
        child_joint: Optional[List[gJoint]] = [],
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name=name, dof=6, id=id, parent_jnt=parent_joint, 
                         child_jnt=child_joint, origin=origin, dtype=dtype, device=device)
    
    @staticmethod
    def _check_input(angle: torch.Tensor):
        if angle.ndim == 1:
            angle = angle.view(-1, 1)
        if angle.ndim != 2:
            raise ValueError("The joint angle must be a vector.")
        return angle

    # parent 에서본 translation 업데이트
    def translation(self, trn: torch.Tensor) -> torch.Tensor:
        FreeJoint._check_input(trn)
        return self._translation_impl(trn.view(-1, 1))
    
    def _translation_impl(self,trn: torch.Tensor):
        self.lmat[:, :, 3] = trn.view(1,-1)

    # default 기준 rotation
    def _rotation_impl(self, exp: torch.Tensor) -> torch.Tensor:
        return SO3.exp(exp)

    def rotation(self, angle: torch.Tensor) -> torch.Tensor:
        return self._rotation_impl(angle)
 
    def _relative_pose_impl(self, angle: torch.Tensor) -> torch.Tensor:
        rot = self.rotation(angle)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3] @ rot
        ret[:, :, 3] = self.origin[:, :, 3]
        return ret
    
    def relative_pose(self, angle: Optional[torch.Tensor] = None) -> torch.Tensor:
        #print("update local rotation")
        angle = FreeJoint._check_input(angle)
        self._lmat = self._relative_pose_impl(angle)
        
        return self._lmat
    
    # update gJoint world matrix
    def update_kinematicpose(self):
        #print("update world origin matrix")
        if(self.parent == None):
            self._wmat = self.lmat
        else:
            self.wmat = SE3.compose(self.parent.wmat, self.lmat)

        for child in self.child:
            child.update_kinematicpose()

    def getCompactArray(self):
        self._q = SE3.log(self.lmat)
        return self._q

    def _setCompactArray_impl(self,exp):
        ret = exp.new_empty(exp.shape[0], 3, 4)
        ret[:, :, :3] = SO3.exp(exp[:,3:])
        ret[:,:,3] = exp[:,:3]      
        return ret
    
    def setCompactArray(self, exp):
        exp = FreeJoint._check_input(exp)
        self._lmat =  self._setCompactArray_impl(exp)

if __name__ == "__main__":
    print("This will be executed when the module is run directly")
   
    ## TEST ##
    angle = torch.deg2rad(torch.tensor([0, (90), 0],dtype=torch.float32))

    # define joints
    base = BallJoint("base") 
    base2 = BallJoint("base2") 
    base3 = BallJoint("base3") 
    test = RevoluteJoint("RevX",torch.tensor([0, 0, 1],dtype=torch.float32))
    testp = PrismaticJoint("Pris",torch.tensor([1, (1), 0],dtype=torch.float32))
    testpx = PrismaticJointZ("PX")
    pelv = FreeJoint("Pelv")

    angle2 = torch.deg2rad(torch.tensor([90],dtype=torch.float32))

    print(test.relative_pose(angle2))
    print(testp.relative_pose(angle2))
    print(testpx.relative_pose(angle2))

    print(base3.origin)
    print(base3.relative_pose(angle))
    print(base.wmat)

    # add kinematic chain
    child_list=[]
    child_list.append(base)
    pelv._child = child_list
    base.setParent(pelv)

    child_list=[]
    child_list.append(base2)
    child_list.append(testp)
    child_list.append(testpx)

    base._child = child_list
    base2.setParent(base)
    testp.setParent(base)
    testpx.setParent(base)

    child_list=[] 
    child_list.append(base3)
    child_list.append(test)
    base2._child = child_list
    base3.setParent(base2)
    test.setParent(base2)

    print(pelv.origin)
    pelv.translation(torch.tensor([0,1,0], dtype=torch.float32))

    # update poses
    pelv.update_kinematicpose()
    print(base.wmat)
    print(base3.wmat)

    print(pelv.setCompactArray(torch.tensor([[0,1,0,0,1.57,0]], dtype=torch.float32)))
    base3.setCompactArray(torch.tensor([0,1.45,0],dtype=torch.float32))
    pelv.update_kinematicpose()
   
    print(pelv.getCompactArray())
    print(base3.getCompactArray())
    print(f"test_{test.getCompactArray()}")
    print(testp.getCompactArray())
    print(testpx.getCompactArray())
    print(testpx._wmat[:,:,3])