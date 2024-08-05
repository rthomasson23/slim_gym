"""Teleoperate robot with keyboard or SpaceMouse.

***Choose user input option with the --device argument***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with macOS, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

    SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
    We used the SpaceMouse Wireless in our experiments. The paper below used the same device
    to collect human demonstrations for imitation learning.

    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018

    Note:
        This current implementation only supports macOS (Linux support can be added).
        Download and install the driver before running the script:
            https://www.3dconnexion.com/service/drivers.html

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).

    Notes:
        OSC also tends to be more computationally efficient since IK relies on the backend pybullet IK solver.


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "TwoArmPegInHole", "NutAssembly", etc.

    --robots: Robot(s) with which to perform the task. Can be any in
        {"Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e", "Baxter"}. Note that the environments include sanity
        checks, such that a "TwoArm..." environment will only accept either a 2-tuple of robot names or a single
        bimanual robot name, according to the specified configuration (see below), and all other environments will
        only accept a single single-armed robot name

    --config: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies the robot
        configuration desired for the task. Options are {"bimanual", "single-arm-parallel", and "single-arm-opposed"}

            -"bimanual": Sets up the environment for a single bimanual robot. Expects a single bimanual robot name to
                be specified in the --robots argument

            -"single-arm-parallel": Sets up the environment such that two single-armed robots are stationed next to
                each other facing the same direction. Expects a 2-tuple of single-armed robot names to be specified
                in the --robots argument.

            -"single-arm-opposed": Sets up the environment such that two single-armed robots are stationed opposed from
                each other, facing each other from opposite directions. Expects a 2-tuple of single-armed robot names
                to be specified in the --robots argument.

    --arm: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies which of the
        multiple arm eef's to control. The other (passive) arm will remain stationary. Options are {"right", "left"}
        (from the point of view of the robot(s) facing against the viewer direction)

    --switch-on-grasp: Exclusively applicable and only should be specified for "TwoArm..." environments. If enabled,
        will switch the current arm being controlled every time the gripper input is pressed

    --toggle-camera-on-grasp: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc

    For two-arm bimanual environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Baxter --config bimanual --arm left --controller osc

    For two-arm multi single-arm robot environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config single-arm-parallel --controller osc


"""
import csv
import argparse
import os

import numpy as np
import time

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action, input2actionOLD
from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper

import mujoco_py    
import glfw
from mujoco_py.generated import const
from scipy.spatial.transform import Rotation

import pygame
import sys

def euler2mat(euler):
    r = Rotation.from_euler('xyz', euler, degrees=False)
    return r.as_matrix()

def quat2mat(quat):
    quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    r =  Rotation.from_quat(quat)
    return r.as_matrix()

def save_data(file_path, subject_name, task, robot, trial, success, success_time, disturbance="", num_fallen_objects="", reset=""):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow([subject_name, task, robot, trial, success, success_time, disturbance, num_fallen_objects, reset])
        
def prompt_user_for_input(input_text):
     # Initialize Pygame
    pygame.init()

    # Set the width and height of the screen -- ALE
    # screen_width = 1980
    # screen_height = 1160
    # font = pygame.font.Font(None, 50)

    # Set the width and height of the screen -- RACHEL
    screen_width = 3840
    screen_height = 2160
    font = pygame.font.Font(None, 75)

    # Set up the screen
    screen = pygame.display.set_mode((screen_width, screen_height))

    # Set the caption of the screen
    pygame.display.set_caption("User Input Prompt")

    # Fill the screen with black color
    screen.fill((0, 0, 0))

    # Display text prompting the user for input
    text_surface = font.render(input_text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(text_surface, text_rect)

    # Update the display
    pygame.display.flip()

    # Event loop to handle user input
    input_received = False
    while not input_received:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                else:
                    # Handle user input here
                    user_input = pygame.key.name(event.key)
                    input_received = True

    # Close the Pygame window after the user presses a key
    pygame.quit()

if __name__ == "__main__":

    ''' 
    ADD THE USER NAME HERE!!!!
    '''
    subject_name = "davinci"
    hand = "right"
    device = 'oculus'

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-  camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--device", type=str, default = device)
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--directory", type=str, default="robosuite/data")
    args = parser.parse_args()


    # Randomly order the tasks
    tasks = ["ConstrainedReorient", "Bookshelf", "DrawerPick"]
    np.random.shuffle(tasks)

    task_names = {}
    task_names["ConstrainedReorient"] = "Cabinet Reorientation"
    task_names["Bookshelf"] = "Cabinet Pick"
    task_names["DrawerPick"] = "Box Pick"
    task_names["DrawerPickTrain"] = "Box Pick Training"
    task_names["BookshelfTrain"] = "Cabinet Pick Training"
    task_names["ConstrainedReorientTrain"] = "Cabinet Reorientation Training"
    task_names["Train"] = "Initial Training"
    
    train_tasks = ["Train", "BookshelfTrain", "ConstrainedReorientTrain", "DrawerPickTrain"]

    tasks = train_tasks + tasks

    # Import controller config
    if device == "keyboard":
        controller_name = "OSC_POSE"
    else:
        controller_name = "OSC_POSE_ABS"   

    # Get controller cronfig
    controller_config = load_controller_config(default_controller=controller_name)

    # Go through each of the tasks
    for task in tasks:

        # Prompt user to begin the task
        prompt_user_for_input(f"The next task: {task_names[task]}. Check out the task summary, then press any key to continue.")
       
        # Randomly order the robot options
        robots = ["Panda", "PandaWrist", "PandaSSLIM"]
       
        if "Train" not in task:
            np.random.shuffle(robots)

        for robot, ind in zip(robots, range(len(robots))):
            # Prompt the user for input, except for the first robot
            if ind != 0:
                prompt_user_for_input("Press any key to begin the next scenerio.")

            if task == "DrawerPick" and robot == "PandaSSLIM":
                initial_placement = np.array([-0.037, -1.025, 0.123, -1.996, -0.083, 0.998, -2.412, 1.290, 1.447])
            elif task == "DrawerPickTrain" and robot == "PandaSSLIM":
                initial_placement = np.array([-0.037, -1.025, 0.123, -1.996, -0.083, 0.998, -2.412, 1.290, 1.447])
            elif task == "DrawerPick" and robot == "PandaWrist":
                initial_placement = np.array([-0.037, -1.025, 0.123, -1.996 ,-0.083, 0.998, -2.412, 1.290, 1.447])
            elif task == "DrawerPickTrain" and robot == "PandaWrist":
                initial_placement = np.array([-0.037, -1.025, 0.123, -1.996 ,-0.083, 0.998, -2.412, 1.290, 1.447])
            elif task == "DrawerPick" and robot == "Panda":
                initial_placement = np.array([-0.037, -1.025, 0.123, -1.996 ,-0.083, 0.998, -2.412])
            elif task == "DrawerPickTrain" and robot == "Panda":
                initial_placement = np.array([-0.037, -1.025, 0.123, -1.996 ,-0.083, 0.998, -2.412])
            else:
                initial_placement = None

            print(f"Starting task: {task} with robot: {robot}")

            # Create argument configuration
            config = {
                "env_name": task,
                "robots": robot,
                "controller_configs": controller_config,
            }

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
                initial_placement=initial_placement,
            )


            # # Wrap this environment in a visualization wrapper
            # env = VisualizationWrapper(env, indicator_configs=None)

            # Wrap this environment in a data collection wrapper
            data_directory = args.directory + "/" + subject_name + "/" + task + "/" + robot + "/"
            if task != "Train" and task != "DrawerPickTrain" and task != "BookshelfTrain" and task != "ConstrainedReorientTrain":
                env = DataCollectionWrapper(env, data_directory)

            # Setup printing options for numbers
            np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

            # initialize device
            if args.device == "keyboard":
                from robosuite.devices import Keyboard
                device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
                env.viewer.add_keypress_callback("any", device.on_press)
                env.viewer.add_keyup_callback("any", device.on_release)
                env.viewer.add_keyrepeat_callback("any", device.on_press)
            elif args.device == "spacemouse":
                from robosuite.devices import SpaceMouse
                use_robotiq = False if (robot == "PandaSSLIM" or robot == "PandaSSLIMOG") else True
                drawer = False if ("DrawerPick" not in task) else True
                device = SpaceMouse(task, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity, use_robotiq=use_robotiq, drawer=drawer)
                env.viewer.add_keypress_callback("any", device.on_press)
                env.viewer.add_keyrepeat_callback("any", device.on_press)
            elif args.device == "oculus":
                from robosuite.devices import Oculus
                use_robotiq = False if (robot == "PandaSSLIM" or robot == "PandaSSLIMOG") else True
                drawer = False if ("DrawerPick" not in task) else True
                device = Oculus(env, task, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity, use_robotiq=use_robotiq, drawer=drawer, hand=hand)
                env.viewer.add_keypress_callback("any", device.on_press)
                env.viewer.add_keyrepeat_callback("any", device.on_press)
            # else:
            #     raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")


            # Set up a CSV file for the user to save the databbbb
            file_path = "robosuite/data/" + subject_name + ".csv"

            if not os.path.exists(file_path):
                # Column labels
                column_labels = ["subject_name", "task", "robot", "trial", "success", "success_time", "disturbance", "num_fallen_objects", "reset"]
                with open(file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(column_labels)
            try:
                while True:

                    # Reset the environment
                    obs = env.reset()

                    # Setup rendering
                    cam_id = 0
                    num_cam = len(env.sim.model.camera_names)
                    env.render()

                    # Initialize variables that should the maintained between resets
                    last_grasp = 0

                    # Initialize device control
                    device.start_control()

                    while True:

                        # Set active robot
                        active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

                        # Get the newest action
                        action, grasp = input2action(device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config)

                        # If action is none, then this a reset so we should break
                        if action is None:
                            break

                        # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
                        # toggle arm control and / or camera viewing angle if requested
                        if last_grasp < 0 < grasp:
                            if args.switch_on_grasp:
                                args.arm = "left" if args.arm == "right" else "right"
                            if args.toggle_camera_on_grasp:
                                cam_id = (cam_id + 1) % num_cam
                                env.viewer.set_camera(camera_id=cam_id)
                                
                        # Update last grasp
                        last_grasp = grasp

                        # Fill out the rest of the action space if necessary
                        rem_action_dim = env.action_dim - action.size
                        if rem_action_dim > 0:
                            # Initialize remaining action space
                            rem_action = np.zeros(rem_action_dim)
                            # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
                            if args.arm == "right":
                                action = np.concatenate([action, rem_action])
                            elif args.arm == "left":
                                action = np.concatenate([rem_action, action])
                            else:
                                # Only right and left arms supported
                                print(
                                    "Error: Unsupported arm specified -- "
                                    "must be either 'right' or 'left'! Got: {}".format(args.arm)
                                )
                        elif rem_action_dim < 0:
                            # We're in an environment with no gripper action space, so trim the action space to be the action dim
                            action = action[: env.action_dim]

                        # Step through the simulation and render
                        obs, reward, done, info = env.step(action)

                        links2check = ["robot0_link0_collision",  "robot0_link1_collision",  "robot0_link2_collision", "robot0_link3_collision", "robot0_link4_collision", "robot0_link5_collision", "robot0_link6_collision", "robot0_link7_collision", "robot0_forearm_col_0", "robot0_forearm_col_1", "robot0_forearm_col_2", "robot0_forearm_col_3"]
                        
                        for i in range(env.sim.data.ncon):
                            con = env.sim.data.contact[i]
                            bool1 = env.sim.model.geom_id2name(con.geom1) in links2check and env.sim.model.geom_id2name(con.geom2) not in links2check
                            bool2 = env.sim.model.geom_id2name(con.geom2) in links2check and env.sim.model.geom_id2name(con.geom1) not in links2check
                            notgripper1 = "robot0_g1_col" not in env.sim.model.geom_id2name(con.geom1)
                            notgripper2 = "robot0_g1_col" not in env.sim.model.geom_id2name(con.geom2)
                            if (bool1 or bool2) and (notgripper1 and notgripper2):
                                contact_pos = con.pos
                                env.viewer.viewer.add_marker(type=const.GEOM_SPHERE, pos=contact_pos, size=np.array([0.02, 0.02, 0.02]), label='contact', rgba=[0.592, 0.863, 1, .4])

                        check_finger = [ "gripper0_distal_left", "gripper0_distal_right", "gripper0_proximal_left", "gripper0_proximal_right""gripper0_base_mount", "gripper0_base", "gripper0_driver_right", "gripper0_driver_left", "gripper0_follower_left", "gripper0_follower_right", "gripper0_coupler_left", "gripper0_coupler_right", "gripper0_spring_link_left", "gripper0_spring_link_right"]
                        check_drawer = ["drawer_collision"]

                        for i in range(env.sim.data.ncon):
                            cont = env.sim.data.contact[i]
                            bool1 = env.sim.model.geom_id2name(cont.geom1) in check_finger and env.sim.model.geom_id2name(cont.geom2) in check_drawer
                            bool2 = env.sim.model.geom_id2name(cont.geom2) in check_finger and env.sim.model.geom_id2name(cont.geom1) in check_drawer
                            if bool1 or bool2:
                                contact_pos = cont.pos
                                env.viewer.viewer.add_marker(type=const.GEOM_SPHERE, pos=contact_pos, size=np.array([0.02, 0.02, 0.02]), label='contact', rgba=[0.592, 0.863, 1, .4])

                        if task == "ConstrainedReorient":
                            trial_ended, success, success_time, trial = env._check_success()
                            if trial_ended:
                                if success == 2: # True success
                                    env.viewer.viewer.add_marker(type=const.GEOM_ARROW, pos=env.goal_pos, mat=euler2mat([np.pi, 0, 0]), label='', size=[0.01, 0.01, 0.6], rgba=[0, 1, 0, 1])
                                save_data(file_path, subject_name, task, robot, trial, success, success_time, reset=device.oculus_policy.number_of_resets)
                                # print number of resets
                                
                                env.render()
                                print("Here ")
                                time.sleep(3)
                                device._reset_internal_state()
                                device.oculus_policy.reinitialize_policy()
                                device.oculus_policy.number_of_resets = 0
                            else:
                                if success == 1:
                                    env.viewer.viewer.add_marker(type=const.GEOM_ARROW, pos=env.goal_pos, mat=euler2mat([np.pi, 0, 0]), label='', size=[0.01, 0.01, 0.6], rgba=[0.902, .616, .094, .6])
                                else:
                                    env.viewer.viewer.add_marker(type=const.GEOM_ARROW, pos=env.goal_pos, mat=euler2mat([np.pi, 0, 0]), label='', size=[0.01, 0.01, 0.6], rgba=[1, 0, 0, .4])

                        elif task == "ConstrainedReorientTrain":
                            new, success, finish = env._check_success()

                            if new: 
                                time.sleep(1) 
                                device._reset_internal_state()
                                device.oculus_policy.reinitialize_policy()
                                env.render()
                                if finish:
                                    raise Exception("Finish")

                            if success == 2: # True success
                                env.viewer.viewer.add_marker(type=const.GEOM_ARROW, pos=env.goal_pos, mat=euler2mat([np.pi, 0, 0]), label='', size=[0.01, 0.01, 0.6], rgba=[0, 1, 0, 1])
                            elif success == 1:
                                env.viewer.viewer.add_marker(type=const.GEOM_ARROW, pos=env.goal_pos, mat=euler2mat([np.pi, 0, 0]), label='', size=[0.01, 0.01, 0.6], rgba=[0.902, .616, .094, .6])
                            else:
                                env.viewer.viewer.add_marker(type=const.GEOM_ARROW, pos=env.goal_pos, mat=euler2mat([np.pi, 0, 0]), label='', size=[0.01, 0.01, 0.6], rgba=[1, 0, 0, .4])


                        elif task == "Bookshelf":
                            trial_ended, success, success_time, disturbance, num_fallen_objects, trial = env._check_success()
                            if trial_ended:
                                if success: 
                                    env.viewer.viewer.add_marker(type=const.GEOM_LABEL, pos=env.goal_object_pos, label='Success!', size=[1,1,1], rgba=[0, 0, 1, 1])
                                save_data(file_path, subject_name, task, robot, trial, success, success_time, disturbance, num_fallen_objects, reset=device.oculus_policy.number_of_resets)
                                env.render()                                
                                time.sleep(3)
                                device._reset_internal_state()
                                device.oculus_policy.reinitialize_policy()
                                device.oculus_policy.number_of_resets = 0

                        elif task == "BookshelfTrain":
                            new, finish = env._check_success()
                            if new: 
                                time.sleep(1)
                                env.render()
                                device._reset_internal_state()
                                device.oculus_policy.reinitialize_policy()
                            if finish:
                                raise Exception("Finish")
                                
                        elif task == "DrawerPick":
                            trial_ended, success, success_time, trial = env._check_success()
                            if trial_ended: 
                                if success:
                                    env.viewer.viewer.add_marker(type=const.GEOM_LABEL, pos=env.place_object_pos, label='Success!', size=[1,1,1], rgba=[0, 0, 1, 1])
                                save_data(file_path, subject_name, task, robot, trial, success, success_time, reset=device.oculus_policy.number_of_resets)
                                
                                env.render()
                                time.sleep(3)
                                device._reset_internal_state()
                                device.oculus_policy.reinitialize_policy()
                                device.oculus_policy.number_of_resets = 0

                        elif task == "DrawerPickTrain":
                            new, finish = env._check_success()

                            if new: 
                                time.sleep(1)
                                env.render()
                                device._reset_internal_state()
                                device.oculus_policy.reinitialize_policy()
                                if finish:
                                    raise Exception("Finish")

                        elif task == "Train":
                            rot = quat2mat(env.place_object_quat)
                            if env.rot_grasp:
                                rot = np.dot(rot, euler2mat([0, -np.pi / 2, 0]))                           

                        env.render()    
            except:
                env.close()
                continue
