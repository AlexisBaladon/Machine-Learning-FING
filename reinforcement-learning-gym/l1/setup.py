from setuptools import setup, find_packages

setup(
    name="aa22-l1",
    version="0.22.0",
    packages=find_packages(),
    install_requires=[
        "gym>=0.25.1",
        "gym-minigrid",
    ]
)