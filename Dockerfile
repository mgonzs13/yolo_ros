ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-core AS deps

ARG ROS_DISTRO

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies early for better caching
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-rosdep \
    python3-colcon-common-extensions \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create ros2_ws and copy files
WORKDIR /root/ros2_ws
COPY . /root/ros2_ws/src

# Foxy, Galactic and Iron do not ship onnxruntime_vendor. Fetch the released
# CPU runtime from the Humble binary package and use it as a custom prefix.
RUN if [ "${ROS_DISTRO}" = "foxy" ] || [ "${ROS_DISTRO}" = "galactic" ] || [ "${ROS_DISTRO}" = "iron" ]; then \
        bash /root/ros2_ws/src/scripts/fetch_onnxruntime.sh /opt/onnxruntime; \
    fi

# Install ROS dependencies
RUN rosdep init && rosdep update --include-eol-distros
# Upgrade the packages shipped in the base image first
RUN apt update && apt upgrade -y && rosdep install --from-paths src --ignore-src -r -y \
    --dependency-types=buildtool \
    --dependency-types=build \
    --dependency-types=exec \
    && rm -rf /var/lib/apt/lists/*

FROM deps AS builder

ARG ROS_DISTRO

SHELL ["/bin/bash", "-c"]

# Build the workspace
WORKDIR /root/ros2_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && if [ -f /opt/onnxruntime/lib/libonnxruntime.so ]; then \
         colcon build --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime; \
       else \
         colcon build; \
       fi

# Source the ROS 2 setup file
RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Run a default command, e.g., starting a bash shell
CMD ["bash"]
