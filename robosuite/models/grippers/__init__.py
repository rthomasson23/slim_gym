from .gripper_model import GripperModel
from .gripper_factory import gripper_factory
from .gripper_tester import GripperTester

from .panda_gripper import PandaGripper
from .rethink_gripper import RethinkGripper
from .robotiq_85_gripper import Robotiq85Gripper
from .robotiq_85_gripper import Robotiq4Wrist
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper, RobotiqThreeFingerDexterousGripper
from .panda_gripper import PandaGripper
from .jaco_three_finger_gripper import JacoThreeFingerGripper, JacoThreeFingerDexterousGripper
from .robotiq_140_gripper import Robotiq140Gripper
from .wiping_gripper import WipingGripper
from .null_gripper import NullGripper
from .rr_gripper import RRGripper
from .reach_gripper import ReachGripper
from .reach_kinova_gripper import ReachKinovaGripper
from .sslim_gripper import SSLIM
from .sslim_hand import SSLIM_Hand
from .sslim_hand import SSLIM_Hand2
from .robotiq_2f85 import Robotiq_2F85
from .robotiq_2f85 import Robotiq_2F85_wrist


GRIPPER_MAPPING = {
    "RethinkGripper": RethinkGripper,
    "PandaGripper": PandaGripper,
    "JacoThreeFingerGripper": JacoThreeFingerGripper,
    "JacoThreeFingerDexterousGripper": JacoThreeFingerDexterousGripper,
    "WipingGripper": WipingGripper,
    "ReachGripper": ReachGripper,
    "ReachKinovaGripper": ReachKinovaGripper,
    "RRGripper": RRGripper,
    "Robotiq85Gripper": Robotiq85Gripper,
    "Robotiq4Wrist": Robotiq4Wrist,
    "Robotiq140Gripper": Robotiq140Gripper,
    "RobotiqThreeFingerGripper": RobotiqThreeFingerGripper,
    "RobotiqThreeFingerDexterousGripper": RobotiqThreeFingerDexterousGripper,
    "ReachGripper": ReachGripper,
    "ReachKinovaGripper": ReachKinovaGripper,
    "SSLIMGripper": SSLIM,
    "SSLIMHand": SSLIM_Hand,
    "SSLIMHand2": SSLIM_Hand2,
    "Robotiq_2F85": Robotiq_2F85,    # from mujoco menagerie
    "Robotiq_2F85_wrist": Robotiq_2F85_wrist,   # from mujoco menagerie 4 wrist
    None: NullGripper,
}

ALL_GRIPPERS = GRIPPER_MAPPING.keys()
