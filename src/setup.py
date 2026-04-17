from glob import glob
import os

from setuptools import find_packages, setup


package_name = "dominoes"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [os.path.join("resource", package_name)],
        ),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="samdong",
    maintainer_email="samdong@example.com",
    description="ROS2 package for running the domino pickup script.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pickup = dominoes.pickup:main",
        ],
    },
)
