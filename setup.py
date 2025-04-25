from setuptools import setup, find_packages

setup(
    name="dppo",
    version="0.1.0",
    packages=find_packages() + ["homebot_sapien", "env.sapien_utils", "env.utils", "env.articulation"],
    install_requires=[
        "torch",
        "numpy",
        "hydra-core",
        "wandb",
        "sapien",
        "gym",
        "opencv-python",
    ],
) 