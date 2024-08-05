"""Driver class for Oculus controller.

"""

import time
from collections import namedtuple


import numpy as np
import rospy
from sensor_msgs.msg import Joy
import glfw
import sys
import copy 


from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix
from scipy.spatial.transform import Rotation

from oculus_reader.reader import OculusReader

def mat2euler(mat):
    r = Rotation.from_matrix(mat)
    return r.as_euler('xyz', degrees=True)

class OculusPolicy(): 
    """ Runs policy using Oculus controller commands"""
    def __init__(self, starting_pos, starting_ori, task="Bookshelf", hand="right"):
        self.oculus_reader = OculusReader()
        self.demo = False
        self.starting_pos = starting_pos
        self.starting_ori = starting_ori

        self.camera_rot = Rotation.from_euler('xyz', [0, -90, 0], degrees=True)
        self.camera_transform = np.eye(4)
        self.camera_transform[:3,:3] = self.camera_rot.as_matrix()
        self.task = task

        self.number_of_resets = 0
        print('Resets resetted to 0')
        self.reset_pressed = False
        self.send_reset = False
        self.hand = hand

    def get_oculus_state(self):
        poses, buttons = self.oculus_reader.get_transformations_and_buttons()
        while poses == {}: 
            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
        if self.task == "DrawerPick" or self.task == "DrawerPickTrain":
            poses = {key: self.camera_transform @ np.array(poses[key]) for key in poses}
        return poses, buttons
        
    def initialize_policy(self, robot_pos_origin, robot_ori_origin):
        """Initializes the Oculus controller."""
        self.robot_pos_origin = robot_pos_origin
        self.robot_ori_origin = robot_ori_origin 
        poses, buttons = self.get_oculus_state()
        
        self.oculus_pos_origin, self.oculus_ori_origin = self.get_oculus_pose(poses, self.hand)
        self.action_pos, self.action_ori, self.gripper_action = self.get_oculus_action(poses, buttons)

    def reinitialize_policy(self):
        self.robot_pos_origin = self.starting_pos
        self.robot_ori_origin = self.starting_ori
        poses, buttons = self.get_oculus_state()
        
        self.oculus_pos_origin, self.oculus_ori_origin = self.get_oculus_pose(poses, self.hand)
        self.action_pos, self.action_ori, self.gripper_action = self.get_oculus_action(poses, buttons)

    def step(self, robot_pos_origin, robot_ori_origin, pos_sensitivity): 
        poses, buttons = self.get_oculus_state()
        self.send_reset = False
        if self.hand == "right":

            if buttons["RG"] and not buttons["RTr"]: 
                self.action_pos, _, self.gripper_action = self.get_oculus_action(poses, buttons, pos_sensitivity)
                self.robot_ori_origin = robot_ori_origin
                _, self.oculus_ori_origin = self.get_oculus_pose(poses, self.hand)

            elif buttons["RTr"] and not buttons["RG"]:
                _, self.action_ori, self.gripper_action = self.get_oculus_action(poses, buttons, pos_sensitivity)
                self.robot_pos_origin = robot_pos_origin
                self.oculus_pos_origin, _ = self.get_oculus_pose(poses, self.hand)

            elif buttons["RG"] and buttons["RTr"]:
                self.action_pos, self.action_ori, self.gripper_action = self.get_oculus_action(poses, buttons, pos_sensitivity)

            elif buttons["RJ"]:
                if not self.reset_pressed:
                    self.reset_pressed = True
                    self.number_of_resets += 1
                    self.send_reset = True
                    self.pressed_time = time.time()
            else:
                _ , _, self.gripper_action = self.get_oculus_action(poses, buttons, pos_sensitivity)
                self.robot_pos_origin = robot_pos_origin 
                self.robot_ori_origin = robot_ori_origin
                self.oculus_pos_origin, self.oculus_ori_origin = self.get_oculus_pose(poses, self.hand)

        elif self.hand == "left":
            if buttons["LG"] and not buttons["LTr"]: 
                self.action_pos, _, self.gripper_action = self.get_oculus_action(poses, buttons, pos_sensitivity)
                self.robot_ori_origin = robot_ori_origin
                _, self.oculus_ori_origin = self.get_oculus_pose(poses, self.hand)

            elif buttons["LTr"] and not buttons["LG"]:
                _, self.action_ori, self.gripper_action = self.get_oculus_action(poses, buttons, pos_sensitivity)
                self.robot_pos_origin = robot_pos_origin
                self.oculus_pos_origin, _ = self.get_oculus_pose(poses, self.hand)

            elif buttons["LG"] and buttons["LTr"]:
                self.action_pos, self.action_ori, self.gripper_action = self.get_oculus_action(poses, buttons, pos_sensitivity)

            elif buttons["LJ"]:
                print("reset_pressed")
                if not self.reset_pressed:
                    self.reset_pressed = True
                    self.number_of_resets += 1
                    self.send_reset = True
                    self.pressed_time = time.time()
            else:
                _ , _, self.gripper_action = self.get_oculus_action(poses, buttons, pos_sensitivity)
                self.robot_pos_origin = robot_pos_origin 
                self.robot_ori_origin = robot_ori_origin
                self.oculus_pos_origin, self.oculus_ori_origin = self.get_oculus_pose(poses, self.hand)


        if self.reset_pressed:
            curr_time = time.time()
            elapsed_time = curr_time - self.pressed_time
            if elapsed_time > 1:
                self.reset_pressed = False

        return self.action_pos, self.action_ori, self.gripper_action, self.send_reset

    def get_oculus_action(self, poses, buttons, pos_sensitivity=1.0, rot_sensitivity=1.0): 
        """Returns the action from the oculus controller."""
        if self.hand == "right":
            oculus_pos = copy.deepcopy(poses['r'][:3,3])
            indices = [2,0,1]
            action_pos = (oculus_pos[indices] - self.oculus_pos_origin[indices])*pos_sensitivity + self.robot_pos_origin
            
            oculus_ori = copy.deepcopy(poses['r'][:3,:3])
            conv_mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            action_ori = (conv_mat @ oculus_ori) @ (conv_mat @ self.oculus_ori_origin).T @ self.robot_ori_origin

        elif self.hand == "left":
            oculus_pos = copy.deepcopy(poses['l'][:3,3])
            indices = [2,0,1]
            action_pos = (oculus_pos[indices] - self.oculus_pos_origin[indices])*pos_sensitivity + self.robot_pos_origin
            
            oculus_ori = copy.deepcopy(poses['l'][:3,:3])
            conv_mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            action_ori = (conv_mat @ oculus_ori) @ (conv_mat @ self.oculus_ori_origin).T @ self.robot_ori_origin

        gripper_action = np.array([0,0])

        if self.hand == "right":
            if buttons["B"]:
                gripper_action[0] = 1
            if buttons["A"]:
                gripper_action[1] = 1
        elif self.hand == "left":
            if buttons["X"]:
                gripper_action[0] = 1
            if buttons["Y"]:
                gripper_action[1] = 1
    

        return action_pos, action_ori, gripper_action

    @staticmethod
    def get_oculus_pose(poses, hand="right"): 
        """Returns the oculus pose."""
        if hand == "right":
            oculus_pos_origin = copy.deepcopy(poses['r'][:3,3])
            oculus_ori_origin = copy.deepcopy(poses['r'][:3,:3])
        elif hand == "left":
            oculus_pos_origin = copy.deepcopy(poses['l'][:3,3])
            oculus_ori_origin = copy.deepcopy(poses['l'][:3,:3])

        return oculus_pos_origin, oculus_ori_origin
    
class Oculus(Device):

    def __init__(self, env, task, pos_sensitivity=1.0, rot_sensitivity=1.0, use_robotiq=False, drawer=False, hand="right"):
        self.env = env

        self.hand = hand
        
        self.starting_pos = np.array(env.sim.data.get_site_xpos('gripper0_grip_site'))
        self.starting_ori = np.array(env.sim.data.get_site_xmat('gripper0_grip_site'))
        self.oculus_policy = OculusPolicy(self.starting_pos, self.starting_ori, task=task, hand=self.hand)   ### ALE
        self.camera_rot = self.oculus_policy.camera_rot.as_matrix()
        self.init_pos = np.array(env.sim.data.get_site_xpos('gripper0_grip_site'))
        self.init_ori = np.array(env.sim.data.get_site_xmat('gripper0_grip_site'))
        self.oculus_policy.initialize_policy(self.init_pos, self.init_ori)

        # turn this value to true if using Robotiq gripper, false if using SSLIM
        self.use_robotiq = use_robotiq
        self.symmetric = True
        self.use_ori = True
        self.task = task

        self.alpha = 0.04
        self.alpha2 = 0.02

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0



        if self.use_robotiq:
            self.dq = [0]
        else:
            self.dq = np.zeros(7)

        if self.task == "Bookshelf" or self.task == "BookshelfTrain":
            print("Bookshelf task")
            self.thumb_pos = -0.6
        elif self.task == "DrawerPick" or self.task == "DrawerPickTrain":
            print("Drawer task")
            self.thumb_pos = 1
        elif self.task == "ConstrainedReorient" or self. task == "ConstrainedReorientTrain":
            print("Constrained Reorient task")
            self.thumb_pos = 0.6
        else:
            print("Train task")
            self.thumb_pos = 0
            
        
        self.pinch = False
        self.link_thumb = True

        self._display_controls()

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        # self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False
        self._buttons = [0.0, 0.0]
        self.reset_robot = False

        self.last_button_state = [0, 0]

        self.last_clicked = [-1, -1]


    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))


    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """

        self._reset_state = 1
        self._enabled = False

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False
        if self.use_robotiq:
            self.dq = [-1]
            self.dq_actual = [-1]
        else:
            # self.sslim_state = 2
            self.dq = np.zeros(7)
            self.dq_actual = np.zeros(7)

    
    def _reset_device_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False
        if self.use_robotiq:
            self.dq = [-1]
            self.dq_actual = [-1]
        else:
            # self.sslim_state = 2
            self.dq = np.zeros(7)
            self.dq_actual = np.zeros(7)

    def _get_oculus_commands(self):
        self.init_pos = np.array(self.env.sim.data.get_site_xpos('gripper0_grip_site'))
        self.init_ori = np.array(self.env.sim.data.get_site_xmat('gripper0_grip_site'))
        action_pos, action_ori, gripper_action, self.reset_robot = self.oculus_policy.step(self.init_pos, self.init_ori, self.pos_sensitivity)
        self._buttons = gripper_action
        self.x, self.y, self.z = action_pos

        rpy = mat2euler(action_ori)
        self.roll, self.pitch, self.yaw = rpy
        
        # if self.task == "DrawerPick":
        #     temp = self.x
        #     self.x = self.y
        #     self.y = -temp

        #     # build a rotation matrix of -90 degrees around the z axis
        #     rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        #     action_ori = rot @ action_ori @ rot.T
        #     rpy = mat2euler(action_ori)
        #     self.roll, self.pitch, self.yaw = rpy


        self._control = [
            self.x,
            self.y,
            self.z,
            self.roll,
            self.pitch,
            self.yaw,
        ]

    def on_press(self, window, key, scancode, action, mods):
        """
        Key handler for key presses.

        Args:
            window: [NOT USED]
            key (int): keycode corresponding to the key that was pressed
            scancode: [NOT USED]
            action: [NOT USED]
            mods: [NOT USED]
        """
        # controls for moving thumb
        if key == glfw.KEY_Q:
            self.thumb_pos += 0.1
        elif key == glfw.KEY_W:
            self.thumb_pos -= 0.1
        elif key == glfw.KEY_A:
            self.pinch = True
            self.dq[1] = 0
            self.dq[3] = 0
            self.dq[6] = 0

        elif key == glfw.KEY_S:
            self.pinch = False
            self.dq[1] = self.dq[0] * (self.alpha2 / self.alpha)
            self.dq[3] = self.dq[2] * (self.alpha2 / self.alpha)
            if self.link_thumb:
                self.dq[6] = self.dq[5] * (self.alpha2 / self.alpha)

        elif key == glfw.KEY_Z:
            self.link_thumb = False
            self.dq[5] = 0
            self.dq[6] = 0

        elif key == glfw.KEY_X:
            self.link_thumb = True
            self.dq[5] = self.dq[2]
            if self.pinch:
                self.dq[6] = 0
            else:
                self.dq[6] = self.dq[5] * (self.alpha2 / self.alpha)

        elif key == glfw.KEY_B:
            self._reset_internal_state()
            self.oculus_policy.reinitialize_policy()

        elif key == glfw.KEY_DELETE:
            raise ValueError("Exit training!")


    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        self._get_oculus_commands()
        dpos = self.control[:3]
        # roll, pitch, yaw = self.control[3:] * 0.008 * self.rot_sensitivity
        roll, pitch, yaw = self.control[3:] 

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))
        # print(self._buttons)

        if self.reset_robot:
            self.env.robots[0].reset()
            self.oculus_policy.reinitialize_policy()
            self.reset_robot = False

        if self.use_robotiq:
            if self._buttons[1]:
                self.dq[0] = min(1, self.dq[0] + 0.1)
            if self._buttons[0]:
                self.dq[0] = max(-1, self.dq[0] - 0.1)
            self.dq_actual = self.dq

        else:                
            if self._buttons[0]:
                if self.dq[0] <= 0.8:
                    self.dq[0] += self.alpha * self.pos_sensitivity
                    self.dq[2] += self.alpha * self.pos_sensitivity
                    self.dq[5] += self.alpha * self.pos_sensitivity
                    
                    self.dq[1] += self.alpha2 * self.pos_sensitivity
                    self.dq[3] += self.alpha2 * self.pos_sensitivity
                    self.dq[6] += self.alpha2 * self.pos_sensitivity

                self.dq_actual = np.copy(self.dq)
                self.dq_actual[1] = min(self.dq[1], 1.35 - self.dq[0])
                self.dq_actual[3] = min(self.dq[3], 1.35 - self.dq[2])
                self.dq_actual[6] = min(self.dq[6], 1.35 - self.dq[5])

            if self._buttons[1]:
                if self.dq[0] >= -.8:
                    self.dq[0] -= self.alpha * self.pos_sensitivity
                    self.dq[2] -= self.alpha * self.pos_sensitivity

                    self.dq[5] -= self.alpha * self.pos_sensitivity

                    self.dq[1] -= self.alpha2 * self.pos_sensitivity
                    self.dq[3] -= self.alpha2 * self.pos_sensitivity
                    self.dq[6] -= self.alpha2 * self.pos_sensitivity

                self.dq_actual = np.copy(self.dq)
                self.dq_actual[1] = max(self.dq[1], -1.35  - self.dq[0])
                self.dq_actual[3] = max(self.dq[3], -1.35  - self.dq[2])
                self.dq_actual[6] = max(self.dq[6], -1.35 - self.dq[5])

            if not self.symmetric:
                self.dq = np.maximum(self.dq, np.zeros_like(self.dq))
        
            if self.task == "ConstrainedReorient" or self.task == "Train" or self.task == "TrainCabinet" or self.task == "ConstrainedReorientTrain":
                self.dq_actual[4] = np.abs(self.dq_actual[0])*(-1.7) + 1
            elif self.task == "Bookshelf" or self.task == "BookshelfTrain":
                self.dq_actual[4] = np.abs(self.dq_actual[0])*(-1.9) + 1
            elif self.task == "DrawerPick" or self.task == "DrawerPickTrain":
                self.dq_actual[4] = np.abs(self.dq_actual[0])*(-1) + 1


        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            dq=self.dq_actual,
            reset=self._reset_state,
        )
    

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse


        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.


        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0


if __name__ == "__main__":
    oculus = Oculus()
    for i in range(100):
        time.sleep(0.02)
