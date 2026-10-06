ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-core AS deps

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

# Install ROS dependencies
RUN rosdep init && rosdep update --include-eol-distros
# Upgrade the packages shipped in the base image first
RUN apt update && apt upgrade -y && rosdep install --filter-for-installers apt --from-paths src --ignore-src -r -y \
    --dependency-types=buildtool \
    --dependency-types=build \
    && rm -rf /var/lib/apt/lists/*

FROM deps AS builder

SHELL ["/bin/bash", "-c"]

# Build the workspace
WORKDIR /root/ros2_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build

# Source the ROS 2 setup file
RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Run a default command, e.g., starting a bash shell
CMD ["bash"]
