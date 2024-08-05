# read the contents of your README file
from os import path

from setuptools import find_packages, setup


setup(
    name="robosuite",
    packages=[package for package in find_packages() if package.startswith("robosuite")],
    install_requires=[
        "numpy>=1.22",
        "numba>=0.49.1",
        "scipy>=1.2.3",
        "mujoco>=2.3.0",
        "Pillow",
        "opencv-python",
        "pynput",
        "termcolor",
        "mujoco_py",
        "Cython==0.29.36",
        "rospkg",
        "h5py",
        "pygame",
        "pybullet",
        "seaborn"
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="robosuite: A Modular Simulation Framework and Benchmark for Robot Learning",
    author="Alessandra Bernardini",
    url="https://github.com/aleBxrna/slim_user_study.git",
    author_email="",
    long_description_content_type="text/markdown",
)
