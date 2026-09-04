import os
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

package_name = "yolo_ros"


class UvSyncMixin:
    """Run `uv sync` into the installed share directory after install."""

    def uv_sync(self) -> None:
        install_base = getattr(self, "install_base", None) or sys.prefix
        source_dir = getattr(self, "egg_base", None) or os.getcwd()
        project = os.path.join(install_base, "share", package_name)
        pyproject = os.path.join(project, "pyproject.toml")
        if not os.path.exists(pyproject):
            os.makedirs(project, exist_ok=True)
            shutil.copy(os.path.join(source_dir, "pyproject.toml"), pyproject)
        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                project,
                "--no-install-project",
                "--no-dev",
                "--python",
                sys.executable,
            ],
            check=True,
        )


class uv_sync_install(UvSyncMixin, install):
    def run(self) -> None:
        super().run()
        self.uv_sync()


class uv_sync_develop(UvSyncMixin, develop):
    def run(self) -> None:
        super().run()
        self.uv_sync()


setup(
    name=package_name,
    version="4.6.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["pyproject.toml"]),
    ],
    cmdclass={
        "install": uv_sync_install,
        "develop": uv_sync_develop,
    },
    zip_safe=True,
)
