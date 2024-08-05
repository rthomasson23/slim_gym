from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

import argparse

import numpy as np
import time

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper

import mujoco_py    
import glfw
from mujoco_py.generated import const
from scipy.spatial.transform import Rotation

from robosuite.scripts.manipulability_analysis import calculateManipulabilityEllipsoid

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="SingleArmEnv", help="Name of the environment to run")
    parser.add_argument("--robots", nargs="+", type=str, default="PandaWrist", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-  camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--directory", type=str, default="/home/rthom/Documents/Research/TRI/sslim_user_study_data")
    args = parser.parse_args()

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == "ik":
        controller_name = "IK_POSE"
    elif args.controller == "osc":
        controller_name = "OSC_POSE"
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller cronfig
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="cabinetview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Create an array of positions to test using meshgrid
    x_range = [-0.2, 0.3]
    y_range = [-0.3, 0.3]
    z_range = [0.3, 0.8]
    positions = np.mgrid[x_range[0]:x_range[1]:0.1, y_range[0]:y_range[1]:0.1, z_range[0]:z_range[1]:0.1]
    positions = positions.reshape(3, -1).T

    position_index = 0
    
    # do visualization
    for i in range(10000):
        # Get the current end effector position
        eef_pos = env.sim.data.get_site_xpos('gripper0_grip_site')

        # Get the desired end effector position
        desired_pos = positions[position_index]

        # Get the error between current and desired position
        error = desired_pos - eef_pos

        
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
