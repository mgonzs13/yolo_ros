from setuptools import setup

package_name = "yolo_ros"

setup(
    name=package_name,
    version="4.6.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["pyproject.toml"]),
    ],
    zip_safe=True,
)
