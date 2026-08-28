#!/usr/bin/env bash
# Start (or stop) the SpaceMouse Joy stack in the background and return immediately.
# Does not occupy a terminal and does not wait on hz.
#
#   bash /root/catkin_ws/src/hilserl_wa2/scripts/start_spacemouse_joy.sh
#   bash .../start_spacemouse_joy.sh --stop
#
# Host once per login:  DISPLAY=:0 xhost +SI:localuser:root
set -euo pipefail

NODE_LOG="${SPACEMOUSE_NODE_LOG:-/tmp/spacenav_node.log}"
DAEMON_LOG="${SPACEMOUSE_DAEMON_LOG:-/tmp/spacenavd.log}"
CFG="${CFG:-/root/catkin_ws/src/hilserl_wa2/configs/spacemouse/default.yaml}"

STOP=0
for arg in "$@"; do
  case "${arg}" in
    --stop) STOP=1 ;;
    -h|--help)
      echo "usage: $0 [--stop] [config.yaml]" >&2
      exit 2
      ;;
    -*)
      echo "unknown flag: ${arg}" >&2
      exit 2
      ;;
    *) CFG="${arg}" ;;
  esac
done

spacenavd_pid() { pgrep -x spacenavd || true; }
spacenav_node_pid() {
  pgrep -f '/devel/lib/spacenav_node/spacenav_node' \
    || pgrep -f '/lib/spacenav_node/spacenav_node' \
    || true
}

if [[ "${STOP}" == "1" ]]; then
  pkill -f '/devel/lib/spacenav_node/spacenav_node' 2>/dev/null || true
  pkill -f '/lib/spacenav_node/spacenav_node' 2>/dev/null || true
  pkill -x spacenavd 2>/dev/null || true
  sleep 0.2
  echo "SPACEMOUSE_JOY: STOPPED"
  exit 0
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "ROS not sourced. Run hil-actor + setup.bash first." >&2
  exit 1
fi
if [[ -f /root/catkin_ws/devel/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /root/catkin_ws/devel/setup.bash
fi
export DISPLAY="${DISPLAY:-:0}"

echo "config=${CFG}"
echo "DISPLAY=${DISPLAY}"

if [[ -z "$(spacenavd_pid)" ]]; then
  echo "starting spacenavd..."
  # Do not use `spacenavd -d` or `setsid` without -f: both can wait forever here.
  pushd / >/dev/null
  nohup spacenavd </dev/null >"${DAEMON_LOG}" 2>&1 &
  disown || true
  popd >/dev/null
else
  echo "spacenavd already running pid=$(spacenavd_pid | tr '\n' ' ')"
fi

if [[ -z "$(spacenav_node_pid)" ]]; then
  if [[ -z "${ROS_DISTRO:-}" ]]; then
    echo "ROS not sourced; cannot start spacenav_node." >&2
    exit 1
  fi
  echo "starting spacenav_node..."
  : >"${NODE_LOG}"
  nohup rosrun spacenav_node spacenav_node </dev/null >>"${NODE_LOG}" 2>&1 &
  disown || true
else
  echo "spacenav_node already running pid=$(spacenav_node_pid | tr '\n' ' ')"
fi

sleep 1
echo "spacenavd_pid=$(spacenavd_pid | tr '\n' ' ')"
echo "spacenav_node_pid=$(spacenav_node_pid | tr '\n' ' ')"
echo "SPACEMOUSE_JOY: STARTED  (backgrounded; ${NODE_LOG})"
exit 0
