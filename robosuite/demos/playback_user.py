import os
import numpy as np
from glob import glob
import robosuite as suite
from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper


def playback_trajectory(env, ep_dir):
    """Playback data from an episode.

    Args:
        env (MujocoEnv): environment instance to playback trajectory in
        ep_dir (str): The path to the directory containing data for an episode.
    """

    # first reload the model from the xml
    xml_path = os.path.join(ep_dir, "model.xml")
    with open(xml_path, "r") as f:
        env.reset_from_xml_string(f.read())

    state_paths = os.path.join(ep_dir, "state_*.npz")

    # read states back, load them one by one, and render
    t = 0
    for state_file in sorted(glob(state_paths)):
        print(state_file)
        dic = np.load(state_file)
        states = dic["states"]
        for state in states:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.render()
            t += 1
            if t % 100 == 0:
                print(t)


if __name__ == "__main__":

    ''' 
    ADD THE USER NAME HERE!!!!
    '''
    # subject_name = "subject_1"
    
    task = "Bookshelf"
    robot = "Panda"


    # Create argument configuration
    config = {
        "env_name": task,
        "robots": robot,
    }

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

    # Wrap this environment in a data collection wrapper
    # data_directory = "robosuite/data/" + subject_name + "/" + task + "/" + robot + "/" + episode
    data_directory = "robosuite/data/subject_3/Bookshelf/Panda/ep_1709141113_4248776"
    
    if task != "Train":
        env = DataCollectionWrapper(env, data_directory)


    playback_trajectory(env, data_directory)

