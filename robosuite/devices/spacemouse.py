"""Driver class for SpaceMouse controller.


This class provides a driver support to SpaceMouse on macOS.
In particular, we assume you are using a SpaceMouse Wireless by default.


To set up a new SpaceMouse controller:
   1. Download and install driver from https://www.3dconnexion.com/service/drivers.html
   2. Install hidapi library through pip
      (make sure you run uninstall hid first if it is installed).
   3. Make sure SpaceMouse is connected before running the script
   4. (Optional) Based on the model of SpaceMouse, you might need to change the
      vendor id and product id that correspond to the device.


For Linux support, you can find open-source Linux drivers and SDKs online.
   See http://spacenav.sourceforge.net/


"""

import time
from collections import namedtuple


import numpy as np
import rospy
from sensor_msgs.msg import Joy
import glfw


from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])


SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.


    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte


    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.


    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling


    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.


    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte


    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


class SpaceMouse(Device):
    """
    A minimalistic driver class for SpaceMouse with HID library.


    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.


    Args:
        vendor_id (int): HID device vendor id
        product_id (int): HID device product id
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, task, vendor_id=9583, product_id=50735, pos_sensitivity=0.8, rot_sensitivity=1.0, use_robotiq=False, drawer=False):
        # print("Opening SpaceMouse device")
        rospy.init_node("spacemouse_listener", anonymous=True)
        rospy.Subscriber("spacenav/joy", Joy, self._get_spacemouse_commands)

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

        if self.task == "Bookshelf":
            print("Bookshelf task")
            self.thumb_pos = -0.6
        elif self.task == "DrawerPick":
            print("Drawer task")
            self.thumb_pos = 0.2
        elif self.task == "ConstrainedReorient":
            print("Constrained Reorient task")
            self.thumb_pos = -0.6
        elif self.task == "Train":
            print("Train task")
            self.thumb_pos = 0

        self.pinch = False
        self.link_thumb = True

        self._display_controls()

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False
        self._buttons = [0.0, 0.0]

        self.last_button_state = [0, 0]
        self.sslim_grasps = np.zeros([5,7])
        # left full grasp
        self.sslim_grasps[0,:] = np.array([4*np.pi, 4*np.pi, 4*np.pi, 4*np.pi, 1, 4*np.pi, 4*np.pi])
        # left pre-grasp
        self.sslim_grasps[1,:] = np.array([np.pi/8, np.pi/10, np.pi/8, np.pi/10, 1, np.pi/8, np.pi/10])
        # open
        self.sslim_grasps[2,:] = np.array([0, 0, 0, 0, 1, 0, 0])
        # right pre-grasp
        self.sslim_grasps[3,:] = np.array([-np.pi/8, -np.pi/10, -np.pi/8, -np.pi/10, 1, -np.pi/8, -np.pi/10])
        # right full grasp
        self.sslim_grasps[4,:] = np.array([-4*np.pi, -4*np.pi, -4*np.pi, -4*np.pi, 1, -4*np.pi, -4*np.pi])

        self.sslim_state = 2 # open by default

        self.last_clicked = [-1, -1]

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        # print("")
        # print_command("Control", "Command")
        # print_command("Right button", "reset simulation")
        # print_command("Left button (hold)", "close gripper")
        # print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        # print_command("Move mouse vertically", "move arm vertically")
        # print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        # print_command("ESC", "quit")
        # print("")

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
        else:
            # self.sslim_state = 2
            self.dq = np.zeros(7)

    def _get_spacemouse_commands(self, data):
        self._axes = data.axes
        self.last_button_state = self._buttons
        self._buttons = data.buttons

        if self.task == "DrawerPick":
            self.y = -self._axes[0] 
            self.x = self._axes[1] 
            self.z = self._axes[2]
        else:
            self.y = self._axes[1] 
            self.x = self._axes[0] 
            self.z = self._axes[2]

        if self.use_ori:
            if self.task == "DrawerPick":
                self.roll = self._axes[3] * -1
                self.pitch = self._axes[4] 
                self.yaw = self._axes[5] * -1
            else:
                self.roll = self._axes[4]
                self.pitch = self._axes[3]
                self.yaw = self._axes[5] * -1
        else:
            self.roll = 0
            self.pitch = 0
            self.yaw = self._axes[5] * -1
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
        dpos = self.control[:3] * 0.008 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.008 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        if self.use_robotiq:
            if self._buttons[0]:
                self.dq[0] = min(1, self.dq[0] + 0.1)
            if self._buttons[1]:
                self.dq[0] = max(-1, self.dq[0] - 0.1)
        else:
            # # if  we have a new button 0 press
            # if self._buttons[0] == 1:
            #     # check if enough time has passed
            #     t0 = time.time()
            #     if t0 - self.last_clicked[0] > 0.1:
            #         self.sslim_state = max(0, self.sslim_state - 1)

            #     self.last_clicked[0] = t0

            # if self._buttons[1] == 1:
            #     t1 = time.time()
            #     if t1 - self.last_clicked[1] > 0.1:
            #         self.sslim_state = min(4, self.sslim_state + 1)
            #     self.last_clicked[1] = t1

            # # command the joint angles corresponding to the desired state
            # self.dq = self.sslim_grasps[self.sslim_state,:]

            # if self.thumb_pos == 1:
            #     self.dq[4] = 1
            # else:
            #     self.dq[4] = -1

            # print(self.sslim_state)


                
            if self._buttons[0]:
                if self.dq[0] <= 1.5:
                    self.dq[0] += self.alpha * self.pos_sensitivity
                    self.dq[2] += self.alpha * self.pos_sensitivity
                    if self.link_thumb:
                        self.dq[5] += self.alpha * self.pos_sensitivity

                    if not self.pinch:
                        self.dq[1] += self.alpha2 * self.pos_sensitivity
                        self.dq[3] += self.alpha2 * self.pos_sensitivity
                        if self.link_thumb:
                            self.dq[6] += self.alpha2 * self.pos_sensitivity

            if self._buttons[1]:
                if self.dq[0] >= -1.5:
                    self.dq[0] -= self.alpha * self.pos_sensitivity
                    self.dq[2] -= self.alpha * self.pos_sensitivity
                    if self.link_thumb:
                        self.dq[5] -= self.alpha * self.pos_sensitivity

                    if not self.pinch:
                        self.dq[1] -= self.alpha2 * self.pos_sensitivity
                        self.dq[3] -= self.alpha2 * self.pos_sensitivity
                        if self.link_thumb:
                            self.dq[6] -= self.alpha2 * self.pos_sensitivity

            if not self.symmetric:
                self.dq = np.maximum(self.dq, np.zeros_like(self.dq))
            self.dq[4] = self.thumb_pos

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            dq=self.dq,
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
    space_mouse = SpaceMouse()
    for i in range(100):
        print(space_mouse.control, space_mouse.control_gripper)
        time.sleep(0.02)
