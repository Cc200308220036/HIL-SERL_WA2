#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate hil-actor

if [[ -f /opt/ros/noetic/setup.bash ]]; then
    source /opt/ros/noetic/setup.bash
fi

if [[ -f /ros_noetic/catkin_ws/devel/setup.bash ]]; then
    source /ros_noetic/catkin_ws/devel/setup.bash
fi

if [[ -f /root/catkin_ws/devel/setup.bash ]]; then
    source /root/catkin_ws/devel/setup.bash
fi

exec "$@"
