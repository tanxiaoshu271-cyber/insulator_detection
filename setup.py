import glob
import os

from setuptools import find_packages
from setuptools import setup

setup(
    name="AxisAnchor",
    version="1.0",
    author="nxue",
    description="AxisAnchor",
    packages=find_packages(),
    install_requires=[
        "torch", 
        "torchvision",
        "opencv-python",
        "cython",
        "matplotlib",
        "yacs",
        "scikit-image",
        "tqdm",
        "python-json-logger",
        "h5py",
        "shapely",
        "seaborn",
        "easydict",
    ],
    extras_require={
        "dev": [
            "pycolmap",
        ]
    }
)
