import numpy as np
import sapien.core as sapien



class BaseArmSimpleController:
    def __init__(self, robot: sapien.Articulation) -> None:
        self.robot = robot
        mount_link_name = "arm_base"
        ee_link_name = "link_tcp"
        mobile_base_joint_names = ["mobile_rotation", "mobile_translation"]
        arm_joint_names = ["joint%d" % i for i in range(1, 8)]
        self.mobile_base_joint_indices = []
        self.arm_joint_indices = []
        self.finger_joint_indices = []
        self.mount_link_idx: int = None
        self.ee_link_idx: int = None
        for link_idx, link in enumerate(self.robot.get_links()):
            if link.get_name() == mount_link_name:
                self.mount_link_idx = link_idx
            elif link.get_name() == ee_link_name:
                self.ee_link_idx = link_idx
        for joint_idx, joint in enumerate(self.robot.get_active_joints()):
            if joint.get_name() in mobile_base_joint_names:
                self.mobile_base_joint_indices.append(joint_idx)
            elif joint.get_name() in arm_joint_names:
                self.arm_joint_indices.append(joint_idx)
            else:
                self.finger_joint_indices.append(joint_idx)
        self.pmodel = self.robot.create_pinocchio_model()
        self.qmask = np.zeros(self.robot.dof, dtype=bool)
        self.qmask[self.arm_joint_indices] = 1

        self._desired_ee_pose: sapien.Pose = None

    def compute_q_target(
        self,  ee_pose: sapien.Pose, finger_width: float, base_velocity: np.ndarray = np.zeros(2, dtype=np.float32)
    ):
        """
        base_velocity: shape (2,), yaw velocity and forward velocity
        ee_pose: desired ee pose in world frame
        """
        # cur_mount_pose = self.robot.get_links()[self.mount_link_idx].get_pose()
        cur_base_pose = self.robot.get_pose()
        initial_q = self.robot.get_qpos()
        desired_q = initial_q.copy()
        if len(self.mobile_base_joint_indices) > 0:
            desired_q[self.mobile_base_joint_indices] = base_velocity
    
        # in articulation base frame
        robot_desired_pose = cur_base_pose.inv().transform(ee_pose)
        # print("base_to_desired", robot_desired_pose)
        result, success, error = self.pmodel.compute_inverse_kinematics(
            self.ee_link_idx,
            robot_desired_pose,
            initial_q,
            self.qmask,
        )
        if success:
            desired_q[self.arm_joint_indices] = result[self.arm_joint_indices]
        else:
            # print("IK fails")
            pass
            # desired_q[self.arm_joint_indices] = result[self.arm_joint_indices]
            # self.pmodel.compute_forward_kinematics(result)
            # print("forward pose", self.pmodel.get_link_pose(self.ee_link_idx), "desired_pose", robot_desired_pose)
            # print("error", error, "result", result[self.arm_joint_indices], "initial_q", initial_q)
            # exit()
        desired_q[self.finger_joint_indices] = finger_width
        return desired_q


class ArmSimpleController:
    def __init__(self, robot: sapien.Articulation) -> None:
        self.robot = robot
        mount_link_name = "panda_link0"
        ee_link_name = "panda_grasptarget"
        # mobile_base_joint_names = ["mobile_rotation", "mobile_translation"]
        arm_joint_names = ["panda_joint%d" % i for i in range(1, 8)]
        # self.mobile_base_joint_indices = []
        self.arm_joint_indices = []
        self.finger_joint_indices = []
        self.mount_link_idx: int = None
        self.ee_link_idx: int = None
        for link_idx, link in enumerate(self.robot.get_links()):
            if link.get_name() == mount_link_name:
                self.mount_link_idx = link_idx
            elif link.get_name() == ee_link_name:
                self.ee_link_idx = link_idx
        for joint_idx, joint in enumerate(self.robot.get_active_joints()):
            if joint.get_name() in arm_joint_names:
                self.arm_joint_indices.append(joint_idx)
            else:
                self.finger_joint_indices.append(joint_idx)
        self.pmodel = self.robot.create_pinocchio_model()
        self.qmask = np.zeros(self.robot.dof, dtype=bool)
        self.qmask[self.arm_joint_indices] = 1

        self._desired_ee_pose: sapien.Pose = None

    def compute_q_target(
        self, ee_pose: sapien.Pose, finger_width: float
    ):
        """
        ee_pose: desired ee pose in world frame
        """
        # cur_mount_pose = self.robot.get_links()[self.mount_link_idx].get_pose()
        cur_base_pose = self.robot.get_pose()
        initial_q = self.robot.get_qpos()
        desired_q = initial_q.copy()
        # in articulation base frame
        robot_desired_pose = cur_base_pose.inv().transform(ee_pose)
        # print("base_to_desired", robot_desired_pose)
        result, success, error = self.pmodel.compute_inverse_kinematics(
            self.ee_link_idx,
            robot_desired_pose,
            initial_q,
            self.qmask,
        )
        if success:
            desired_q[self.arm_joint_indices] = result[self.arm_joint_indices]
        else:
            # print("IK fails")
            pass
            # desired_q[self.arm_joint_indices] = result[self.arm_joint_indices]
            # self.pmodel.compute_forward_kinematics(result)
            # print("forward pose", self.pmodel.get_link_pose(self.ee_link_idx), "desired_pose", robot_desired_pose)
            # print("error", error, "result", result[self.arm_joint_indices], "initial_q", initial_q)
            # exit()
        desired_q[self.finger_joint_indices] = finger_width
        return desired_q


class PandaSimpleController:
    def __init__(self, robot: sapien.Articulation) -> None:
        self.robot = robot
        mount_link_name = "panda_link0"
        ee_link_name = "panda_grasptarget"
        # mobile_base_joint_names = ["mobile_rotation", "mobile_translation"]
        arm_joint_names = ["panda_joint%d" % i for i in range(1, 8)]
        # self.mobile_base_joint_indices = []
        self.arm_joint_indices = []
        self.finger_joint_indices = []
        self.mount_link_idx: int = None
        self.ee_link_idx: int = None
        for link_idx, link in enumerate(self.robot.get_links()):
            if link.get_name() == mount_link_name:
                self.mount_link_idx = link_idx
            elif link.get_name() == ee_link_name:
                self.ee_link_idx = link_idx
        for joint_idx, joint in enumerate(self.robot.get_active_joints()):
            if joint.get_name() in arm_joint_names:
                self.arm_joint_indices.append(joint_idx)
            else:
                self.finger_joint_indices.append(joint_idx)
        self.pmodel = self.robot.create_pinocchio_model()
        self.qmask = np.zeros(self.robot.dof, dtype=bool)
        self.qmask[self.arm_joint_indices] = 1

        self._desired_ee_pose: sapien.Pose = None

    def compute_q_target(
        self, ee_pose: sapien.Pose, finger_width: float
    ):
        """
        ee_pose: desired ee pose in world frame
        """
        # cur_mount_pose = self.robot.get_links()[self.mount_link_idx].get_pose()
        cur_base_pose = self.robot.get_pose()
        initial_q = self.robot.get_qpos()
        desired_q = initial_q.copy()
        # in articulation base frame
        robot_desired_pose = cur_base_pose.inv().transform(ee_pose)
        # print("base_to_desired", robot_desired_pose)
        result, success, error = self.pmodel.compute_inverse_kinematics(
            self.ee_link_idx,
            robot_desired_pose,
            initial_q,
            self.qmask,
        )
        if success:
            desired_q[self.arm_joint_indices] = result[self.arm_joint_indices]
            success = True
        else:
            # print("IK fails")
            success = False
            # desired_q[self.arm_joint_indices] = result[self.arm_joint_indices]
            # self.pmodel.compute_forward_kinematics(result)
            # print("forward pose", self.pmodel.get_link_pose(self.ee_link_idx), "desired_pose", robot_desired_pose)
            # print("error", error, "result", result[self.arm_joint_indices], "initial_q", initial_q)
            # exit()
        desired_q[self.finger_joint_indices] = finger_width
        return desired_q, success
