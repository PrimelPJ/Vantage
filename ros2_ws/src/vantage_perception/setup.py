from setuptools import find_packages, setup
import os
from glob import glob

package_name = "vantage_perception"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "onnxruntime"],
    zip_safe=True,
    maintainer="Primel Jayawardana",
    maintainer_email="you@primelj.dev",
    description="Edge/cloud object detection router for ROS2",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "router = vantage_perception.inference_router:main",
        ],
    },
)
