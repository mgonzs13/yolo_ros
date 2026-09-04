import glob
import os
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

package_name = "yolo_ros"


class UvSyncMixin:
    """Create the runtime venv and point the nodes at it."""

    def uv_sync(self) -> None:
        install_base = getattr(self, "install_base", None) or sys.prefix
        source_dir = getattr(self, "egg_base", None) or os.getcwd()
        project = os.path.join(install_base, "share", package_name)
        pyproject = os.path.join(project, "pyproject.toml")
        if not os.path.exists(pyproject):
            os.makedirs(project, exist_ok=True)
            shutil.copy(os.path.join(source_dir, "pyproject.toml"), pyproject)

        venv = os.path.join(project, ".venv")
        pyvenv_cfg = os.path.join(venv, "pyvenv.cfg")

        def has_system_site_packages() -> bool:
            if not os.path.exists(pyvenv_cfg):
                return False
            with open(pyvenv_cfg, encoding="utf-8") as f:
                return "include-system-site-packages = true" in f.read()

        if not has_system_site_packages():
            shutil.rmtree(venv, ignore_errors=True)
            subprocess.run(
                [
                    "uv",
                    "venv",
                    "--python",
                    sys.executable,
                    "--system-site-packages",
                    venv,
                ],
                check=True,
            )

        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                project,
                "--no-install-project",
                "--no-dev",
            ],
            check=True,
        )

        if not has_system_site_packages():
            raise RuntimeError(
                f"The virtual environment in '{venv}' was recreated by "
                "`uv sync` without '--system-site-packages'. "
                "Recreate it manually: "
                f"uv venv --python {sys.executable} --system-site-packages {venv}"
            )

        # point the installed entry point scripts at the venv interpreter
        scripts_dir = os.path.join(install_base, "lib", package_name)
        venv_python = os.path.join(venv, "bin", "python")
        for script in glob.glob(os.path.join(scripts_dir, "*")):
            if os.path.isdir(script):
                continue
            with open(script, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines and lines[0].startswith("#!"):
                lines[0] = f"#!{venv_python}\n"
                with open(script, "w", encoding="utf-8") as f:
                    f.writelines(lines)


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
